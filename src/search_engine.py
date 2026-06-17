"""
Semantic Search Engine
Builds a local vector index using SentenceTransformers + FAISS.
Allows natural-language queries over ingested documents.
No internet or paid APIs required.
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Tuple

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class SemanticSearchEngine:
    """
    Local semantic search over processed documents.

    Uses:
    - SentenceTransformers (all-MiniLM-L6-v2) for embeddings
    - FAISS for fast approximate nearest-neighbor search
    Falls back to TF-IDF keyword search if dependencies unavailable.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self):
        self.model = None
        self.index = None
        self.documents = []   # List of {"filename": ..., "text": ..., "class": ...}
        self.embeddings = None
        self._use_faiss = FAISS_AVAILABLE and SENTENCE_TRANSFORMERS_AVAILABLE

    def _load_model(self):
        """Lazy-load the embedding model."""
        if self.model is None:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise RuntimeError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )
            print(f"  Loading embedding model ({self.MODEL_NAME})...")
            self.model = SentenceTransformer(self.MODEL_NAME)
            print("  Model loaded.")

    def build_index(self, processed_results: dict):
        """
        Build the search index from processed document results.
        processed_results: dict from processor.process_folder()
        """
        self.documents = []
        texts = []

        for filename, data in processed_results.items():
            text = data.get("raw_text", "")
            if not text:
                # Reconstruct a searchable snippet from extracted fields
                text = self._fields_to_text(filename, data)

            self.documents.append({
                "filename": filename,
                "class": data.get("class", "Unknown"),
                "text": text,
                "fields": {k: v for k, v in data.items() if k not in ("class", "raw_text")},
            })
            texts.append(text if text else filename)

        if self._use_faiss:
            self._load_model()
            print("  Encoding documents...")
            self.embeddings = self.model.encode(texts, show_progress_bar=False)
            dim = self.embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(np.array(self.embeddings, dtype=np.float32))
            print(f"  FAISS index built with {len(texts)} documents (dim={dim}).")
        else:
            # TF-IDF fallback
            print("  Using TF-IDF fallback (FAISS/SentenceTransformers not available).")
            self._build_tfidf_index(texts)

    def _fields_to_text(self, filename: str, data: dict) -> str:
        """Convert extracted fields into a searchable text blob."""
        parts = [filename]
        for k, v in data.items():
            if k not in ("class", "raw_text") and v is not None:
                parts.append(f"{k}: {v}")
        return " ".join(str(p) for p in parts)

    def _build_tfidf_index(self, texts: List[str]):
        """TF-IDF fallback index."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        self._tfidf_vectorizer = TfidfVectorizer(stop_words="english")
        self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(texts)
        self._cosine_similarity = cosine_similarity

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        """
        Search documents by semantic meaning.
        Returns top_k results with similarity scores.
        """
        if not self.documents:
            return []

        if self._use_faiss and self.index is not None:
            return self._faiss_search(query, top_k)
        else:
            return self._tfidf_search(query, top_k)

    def _faiss_search(self, query: str, top_k: int) -> List[dict]:
        """FAISS-based semantic search."""
        query_vec = self.model.encode([query])
        distances, indices = self.index.search(
            np.array(query_vec, dtype=np.float32), min(top_k, len(self.documents))
        )
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            doc = self.documents[idx]
            # Convert L2 distance to a 0-1 similarity score
            similarity = float(1 / (1 + dist))
            results.append({
                "filename": doc["filename"],
                "class": doc["class"],
                "score": round(similarity, 4),
                "snippet": doc["text"][:200].replace("\n", " "),
                "fields": doc["fields"],
            })
        return results

    def _tfidf_search(self, query: str, top_k: int) -> List[dict]:
        """TF-IDF cosine similarity fallback search."""
        query_vec = self._tfidf_vectorizer.transform([query])
        sims = self._cosine_similarity(query_vec, self._tfidf_matrix).flatten()
        top_indices = sims.argsort()[::-1][:top_k]
        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            results.append({
                "filename": doc["filename"],
                "class": doc["class"],
                "score": round(float(sims[idx]), 4),
                "snippet": doc["text"][:200].replace("\n", " "),
                "fields": doc["fields"],
            })
        return results

    def save_index(self, path: str):
        """Persist the index to disk."""
        state = {
            "documents": self.documents,
            "embeddings": self.embeddings,
            "use_faiss": self._use_faiss,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        if self._use_faiss and self.index:
            faiss.write_index(self.index, path + ".faiss")
        print(f"  Index saved to {path}")

    def load_index(self, path: str):
        """Load a previously saved index."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.documents = state["documents"]
        self.embeddings = state.get("embeddings")
        self._use_faiss = state.get("use_faiss", False)
        if self._use_faiss:
            self._load_model()
            self.index = faiss.read_index(path + ".faiss")
        else:
            texts = [d["text"] for d in self.documents]
            self._build_tfidf_index(texts)
        print(f"  Index loaded: {len(self.documents)} documents.")
