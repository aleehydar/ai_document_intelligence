#!/usr/bin/env python3
"""
AI Document Intelligence System — CLI
======================================
Usage:
  python main.py process --folder ./dataset
  python main.py search  --query "payments due in January"
  python main.py search  --query "invoice from Acme" --top 3

Run `python main.py --help` for full usage.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add src/ to path so sub-modules resolve
sys.path.insert(0, str(Path(__file__).parent / "src"))

from processor import process_folder, save_output
from search_engine import SemanticSearchEngine

INDEX_PATH = "search_index.pkl"
OUTPUT_PATH = "output.json"


def cmd_process(args):
    folder = args.folder
    if not os.path.isdir(folder):
        print(f"[ERROR] Folder not found: {folder}")
        sys.exit(1)

    print(f"\n📂 Processing documents in: {folder}")
    results = process_folder(folder)

    # Save output.json
    out_path = args.output or OUTPUT_PATH
    save_output(results, out_path)

    # Build and save search index
    print("\n🔍 Building semantic search index...")
    engine = SemanticSearchEngine()
    engine.build_index(results)
    engine.save_index(INDEX_PATH)

    # Print summary table
    print("\n" + "=" * 60)
    print(f"{'File':<30} {'Class':<15}")
    print("-" * 60)
    for fname, data in results.items():
        print(f"{fname:<30} {data.get('class','?'):<15}")
    print("=" * 60)
    print(f"\n Done. {len(results)} documents processed.")
    print(f"   Output: {out_path}")
    print(f"   Index:  {INDEX_PATH}")


def cmd_search(args):
    if not os.path.exists(INDEX_PATH):
        print("[ERROR] No search index found. Run `python main.py process` first.")
        sys.exit(1)

    print(f"\n🔍 Loading index...")
    engine = SemanticSearchEngine()
    engine.load_index(INDEX_PATH)

    query = args.query
    top_k = args.top or 5
    print(f"   Query: \"{query}\"")
    print(f"   Top-{top_k} results:\n")

    results = engine.search(query, top_k=top_k)
    if not results:
        print("  No results found.")
        return

    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['class']}] {r['filename']}  (score: {r['score']})")
        if r.get("fields"):
            for k, v in r["fields"].items():
                if v is not None:
                    print(f"       {k}: {v}")
        if r.get("snippet"):
            print(f"       Preview: {r['snippet'][:120]}...")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="AI Document Intelligence System — local PDF classification & search"
    )
    sub = parser.add_subparsers(dest="command")

    # process command
    p_proc = sub.add_parser("process", help="Ingest & process PDFs from a folder")
    p_proc.add_argument("--folder", default="./dataset", help="Path to PDF folder")
    p_proc.add_argument("--output", default=None, help="Output JSON path (default: output.json)")

    # search command
    p_search = sub.add_parser("search", help="Search documents by natural language query")
    p_search.add_argument("--query", required=True, help="Natural language search query")
    p_search.add_argument("--top", type=int, default=5, help="Number of results to return")

    args = parser.parse_args()

    if args.command == "process":
        cmd_process(args)
    elif args.command == "search":
        cmd_search(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
