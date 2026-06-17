"""
Document Intelligence — Local Web Server
Run:  python server.py
Then open http://localhost:5000
"""

import json
import os
import sys
import tempfile
import threading
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

sys.path.insert(0, str(Path(__file__).parent / "src"))
from processor import (
    classify_document,
    extract_invoice_fields,
    extract_resume_fields,
    extract_text_from_file,
    extract_utility_fields,
    save_output,
)
from search_engine import SemanticSearchEngine

INDEX_PATH = "search_index.pkl"

app = Flask(__name__, static_folder="ui", static_url_path="")

# ── in-memory job state ──────────────────────────────────────────────────────
_job = {
    "running": False,
    "total": 0,
    "done": 0,
    "current": "",
    "results": None,
    "error": None,
}
_job_lock = threading.Lock()


# ── helpers ──────────────────────────────────────────────────────────────────

def _classify_one(pdf_path: str) -> dict:
    filename = Path(pdf_path).name
    text = extract_text_from_file(pdf_path)
    doc_class = classify_document(text, filename)
    record = {"class": doc_class}
    if doc_class == "Invoice":
        record.update(extract_invoice_fields(text))
    elif doc_class == "Resume":
        record.update(extract_resume_fields(text))
    elif doc_class == "Utility Bill":
        record.update(extract_utility_fields(text))
    return record


def _run_folder(folder: str):
    try:
        docs = sorted(
            list(Path(folder).glob("*.pdf")) + list(Path(folder).glob("*.txt"))
        )
        with _job_lock:
            _job["total"] = len(docs)
            _job["done"] = 0
            _job["results"] = None
            _job["error"] = None

        results = {}
        for doc in docs:
            with _job_lock:
                _job["current"] = doc.name
            try:
                text = extract_text_from_file(str(doc))
                doc_class = classify_document(text, doc.name)
                record = {"class": doc_class, "raw_text": text}
                if doc_class == "Invoice":
                    record.update(extract_invoice_fields(text))
                elif doc_class == "Resume":
                    record.update(extract_resume_fields(text))
                elif doc_class == "Utility Bill":
                    record.update(extract_utility_fields(text))
                results[doc.name] = record
            except Exception as exc:
                results[doc.name] = {"class": "Unclassifiable", "raw_text": "", "error": str(exc)}
            with _job_lock:
                _job["done"] += 1

        out_path = Path(folder) / "output.json"
        save_output(results, str(out_path))
        save_output(results, str(Path(__file__).parent / "output.json"))

        engine = SemanticSearchEngine()
        engine.build_index(results)
        engine.save_index(str(Path(__file__).parent / INDEX_PATH))

        clean = {k: {kk: vv for kk, vv in v.items() if kk != "raw_text"} for k, v in results.items()}
        with _job_lock:
            _job["running"] = False
            _job["current"] = ""
            _job["results"] = clean
    except Exception as exc:
        with _job_lock:
            _job["running"] = False
            _job["current"] = ""
            _job["error"] = str(exc)


# ── routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("ui", "index.html")


@app.route("/api/process", methods=["POST"])
def api_process():
    folder = (request.json or {}).get("folder", "").strip()
    if not folder or not os.path.isdir(folder):
        return jsonify({"error": f"Folder not found: {folder}"}), 400
    pdfs = list(Path(folder).glob("*.pdf")) + list(Path(folder).glob("*.txt"))
    if not pdfs:
        return jsonify({"error": "No PDF or text files found in that folder."}), 400

    with _job_lock:
        if _job["running"]:
            return jsonify({"error": "A job is already running."}), 409
        _job["running"] = True

    t = threading.Thread(target=_run_folder, args=(folder,), daemon=True)
    t.start()
    return jsonify({"ok": True, "total": len(pdfs)})


@app.route("/api/status")
def api_status():
    with _job_lock:
        return jsonify(dict(_job))


@app.route("/api/dataset")
def api_dataset():
    folder = request.args.get("folder", "./dataset").strip() or "./dataset"
    folder_path = Path(folder)
    if not folder_path.is_dir():
        return jsonify({"error": f"Folder not found: {folder}"}), 400

    pdfs = sorted(p.name for p in list(folder_path.glob("*.pdf")) + list(folder_path.glob("*.txt")))
    results = None
    output_path = folder_path / "output.json"
    if output_path.exists():
        try:
            with open(output_path, encoding="utf-8") as fh:
                results = json.load(fh)
        except (json.JSONDecodeError, OSError):
            results = None

    return jsonify({
        "folder": folder,
        "pdf_count": len(pdfs),
        "pdfs": pdfs,
        "results": results,
        "has_results": results is not None,
    })


@app.route("/api/search", methods=["POST"])
def api_search():
    data = request.json or {}
    query = (data.get("query") or "").strip()
    top_k = int(data.get("top", 5))
    if not query:
        return jsonify({"error": "Query is required."}), 400

    index_path = Path(__file__).parent / INDEX_PATH
    if not index_path.exists():
        return jsonify({"error": "No search index found. Process documents first."}), 404

    engine = SemanticSearchEngine()
    engine.load_index(str(index_path))
    results = engine.search(query, top_k=top_k)
    return jsonify({"query": query, "results": results})


@app.route("/api/classify", methods=["POST"])
def api_classify():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    f = request.files["file"]
    if not f.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported."}), 400

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        f.save(tmp.name)
        tmp_path = tmp.name

    try:
        record = _classify_one(tmp_path)
        record["filename"] = f.filename
        return jsonify(record)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    print("\n  Document Intelligence UI")
    print("  → http://localhost:5000\n")
    app.run(debug=False, port=5000)
