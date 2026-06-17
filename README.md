# AI Document Intelligence System

A **fully local** document processing pipeline that ingests PDF/text files, classifies them, extracts structured fields, and supports semantic search — with no paid or hosted AI APIs.

---

## Features

| Capability | Description |
|---|---|
| **Ingestion** | Reads all `.pdf` and `.txt` files from a folder |
| **Classification** | Invoice, Resume, Utility Bill, Other, Unclassifiable |
| **Extraction** | Type-specific structured fields via regex patterns |
| **Semantic Search** | SentenceTransformers + FAISS local vector index |
| **Interfaces** | CLI (`main.py`) and optional Web UI (`server.py`) |

---

## Installation

### 1. Create a virtual environment (recommended)

```bash
cd ai_document_intelligence1
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**First run only:** SentenceTransformers downloads the `all-MiniLM-L6-v2` model (~90 MB) from Hugging Face. After that, everything runs offline.

---

## Usage

### CLI — Process documents

Ingest all PDFs/text files, classify, extract fields, and build the search index:

```bash
python main.py process --folder ./dataset
```

This produces:
- `output.json` — classifications and extracted fields
- `search_index.pkl` (+ `.faiss`) — semantic search index

### CLI — Semantic search

```bash
python main.py search --query "payments due in January"
python main.py search --query "software engineer with 5 years experience" --top 3
```

### Web UI (optional)

```bash
python server.py
```

Open **http://localhost:5000** to:
- Scan a folder with live progress
- Filter results by document class
- Classify a single uploaded PDF
- Export results as JSON

---

## Output format

`output.json` maps each filename to its class and extracted fields:

```json
{
  "invoice_1.pdf": {
    "class": "Invoice",
    "invoice_number": "1001",
    "date": "2025-06-16",
    "company": "Pioneer Ltd",
    "total_amount": 2073.0
  },
  "resume_1.pdf": {
    "class": "Resume",
    "name": "John Doe",
    "email": "john.doe@example.com",
    "phone": "+1-555-799-6125",
    "experience_years": 5
  }
}
```

---

## Approach & Architecture

```
dataset/*.pdf
      │
      ▼
┌─────────────────┐
│  Text Extraction │  pypdf (+ pdfminer fallback)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Classification  │  Keyword/regex scoring heuristics
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Field Extraction │  Regex patterns per document type
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Semantic Index   │  SentenceTransformers → FAISS
└─────────────────┘
```

### Classification

Documents are scored against keyword lists for each category. Filename hints (e.g. `invoice_1.pdf`) provide additional signal. A confidence margin threshold prevents ambiguous classifications. Empty or unreadable files become **Unclassifiable**.

### Field extraction

| Class | Fields |
|---|---|
| Invoice | `invoice_number`, `date`, `company`, `total_amount` |
| Resume | `name`, `email`, `phone`, `experience_years` |
| Utility Bill | `account_number`, `date`, `usage_kwh`, `amount_due` |
| Other / Unclassifiable | class only |

### Semantic search

- **Embeddings:** `all-MiniLM-L6-v2` via SentenceTransformers (384-dim vectors)
- **Index:** FAISS `IndexFlatL2` for exact nearest-neighbor search
- **Fallback:** TF-IDF + cosine similarity (scikit-learn) if FAISS/ST unavailable

---

## Libraries used

| Library | Purpose |
|---|---|
| [pypdf](https://pypdf.readthedocs.io/) | Primary PDF text extraction |
| [pdfminer.six](https://pdfminersix.readthedocs.io/) | Fallback PDF extraction |
| [SentenceTransformers](https://www.sbert.net/) | Local embedding model |
| [FAISS](https://github.com/facebookresearch/faiss) | Vector similarity search |
| [scikit-learn](https://scikit-learn.org/) | TF-IDF fallback search |
| [Flask](https://flask.palletsprojects.com/) | Optional web UI/API |
| [NumPy](https://numpy.org/) | Vector operations |

---

## Project structure

```
ai_document_intelligence1/
├── main.py              # CLI entry point
├── server.py            # Optional Flask web UI
├── requirements.txt
├── output.json          # Generated results
├── search_index.pkl     # Generated search index
├── src/
│   ├── processor.py     # Ingestion, classification, extraction
│   └── search_engine.py # Semantic search engine
├── ui/
│   └── index.html       # Browser UI
└── dataset/             # Sample PDFs (20 documents)
```

---

## Offline operation

After the initial model download, the system runs entirely offline. No OpenAI, Claude, Gemini, or other hosted AI services are used.
