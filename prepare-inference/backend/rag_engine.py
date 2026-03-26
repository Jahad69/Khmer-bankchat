"""
rag_engine.py
-------------
RAG (Retrieval-Augmented Generation) engine using FAISS.

• Loads a CSV knowledge base with 'Question' and 'Answer' columns.
• Encodes questions with a multilingual sentence-transformer.
• Retrieves top-k similar Q&A pairs for a user query.
• Uses khmernltk for Khmer word segmentation before encoding.
"""

import os
import logging
from pathlib import Path
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"


class RAGEngine:
    def __init__(self):
        self._loaded = False
        self.kb_questions: list[str] = []
        self.kb_answers: list[str] = []
        self.faiss_index = None
        self.embedding_model = None
        self._khmer_tokenizer_available = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, csv_path: str) -> dict:
        """Load knowledge base from CSV and build FAISS index."""
        import pandas as pd
        from sentence_transformers import SentenceTransformer
        import faiss

        if not os.path.exists(csv_path):
            return {"success": False, "error": f"File not found: {csv_path}"}

        try:
            df = pd.read_csv(csv_path)
            if "Question" not in df.columns or "Answer" not in df.columns:
                return {"success": False, "error": "CSV must have 'Question' and 'Answer' columns"}

            self.kb_questions = df["Question"].fillna("").tolist()
            self.kb_answers = df["Answer"].fillna("").tolist()

            # Try loading khmer tokenizer
            try:
                from khmernltk import word_tokenize as khmer_tokenize
                self._khmer_tokenize = khmer_tokenize
                self._khmer_tokenizer_available = True
                logger.info("✅ khmernltk loaded for Khmer segmentation")
            except ImportError:
                logger.warning("⚠️ khmernltk not installed – skipping segmentation")
                self._khmer_tokenizer_available = False

            # Encode questions
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

            segmented = [self._segment(q) for q in self.kb_questions]
            embeddings = self.embedding_model.encode(
                segmented, show_progress_bar=True, batch_size=64
            ).astype("float32")

            # Build FAISS index
            dim = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatL2(dim)
            self.faiss_index.add(embeddings)

            self._loaded = True
            logger.info(f"✅ RAG index built: {len(self.kb_questions)} Q&A pairs")
            return {
                "success": True,
                "num_pairs": len(self.kb_questions),
                "categories": df["Category"].nunique() if "Category" in df.columns else None,
            }
        except Exception as e:
            logger.error(f"RAG load failed: {e}")
            return {"success": False, "error": str(e)}

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """Retrieve the top-k most relevant Q&A pairs for a query."""
        if not self._loaded:
            return []

        segmented = self._segment(query)
        query_emb = self.embedding_model.encode([segmented]).astype("float32")
        distances, indices = self.faiss_index.search(query_emb, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.kb_questions):
                results.append({
                    "question": self.kb_questions[idx],
                    "answer": self.kb_answers[idx],
                    "distance": float(dist),
                })
        return results

    def build_context(self, query: str, top_k: int = 3) -> str:
        """Return a formatted context string ready to inject into the prompt."""
        hits = self.retrieve(query, top_k)
        if not hits:
            return ""
        parts = [f"Q: {h['question']}\nA: {h['answer']}" for h in hits]
        return "\n\n".join(parts)

    @property
    def is_ready(self) -> bool:
        return self._loaded

    def status(self) -> dict:
        return {
            "loaded": self._loaded,
            "num_pairs": len(self.kb_questions),
            "khmer_segmentation": self._khmer_tokenizer_available,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _segment(self, text: str) -> str:
        """Segment Khmer text using khmernltk if available."""
        if self._khmer_tokenizer_available:
            try:
                return " ".join(self._khmer_tokenize(text))
            except Exception:
                pass
        return text


# Global singleton
rag_engine = RAGEngine()
