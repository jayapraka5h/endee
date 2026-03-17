"""
embeddings.py — Embedding pipeline using sentence-transformers
Handles dense vector embedding + sparse TF-IDF vector computation
"""

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

# Global model instances (lazy loaded)
_embed_model = None
_tfidf_vectorizer = None
_tfidf_fitted = False

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        logger.info(f"Loading embedding model: {MODEL_NAME}")
        _embed_model = SentenceTransformer(MODEL_NAME)
        logger.info("Embedding model loaded.")
    return _embed_model


def embed_text(text: str) -> List[float]:
    """Embed a single text string into a dense vector."""
    model = get_embed_model()
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


def batch_embed(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts in batch (more efficient)."""
    model = get_embed_model()
    vectors = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=True)
    return vectors.tolist()


# ─────────────────────────────────────────────
# Sparse TF-IDF vectorization for hybrid search
# ─────────────────────────────────────────────

def fit_tfidf(corpus: List[str]):
    """Fit TF-IDF vectorizer on the corpus. Must be called before compute_sparse_vector."""
    global _tfidf_vectorizer, _tfidf_fitted
    logger.info(f"Fitting TF-IDF on {len(corpus)} documents...")
    _tfidf_vectorizer = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    _tfidf_vectorizer.fit(corpus)
    _tfidf_fitted = True
    logger.info("TF-IDF fitted.")


def compute_sparse_vector(text: str) -> Dict[int, float]:
    """
    Compute sparse TF-IDF vector for a text.
    Returns dict of {term_index: weight} with only non-zero entries.
    """
    global _tfidf_vectorizer, _tfidf_fitted
    if not _tfidf_fitted or _tfidf_vectorizer is None:
        # Return empty sparse vector if not fitted yet
        return {0: 0.001}

    vec = _tfidf_vectorizer.transform([text])
    cx = vec.tocoo()
    sparse = {}
    for col, val in zip(cx.col, cx.data):
        if val > 0:
            sparse[int(col)] = float(val)
    # Endee sparse vector needs at least 1 entry
    if not sparse:
        sparse[0] = 0.001
    return sparse


def sparse_dict_to_endee_format(sparse: Dict[int, float]) -> dict:
    """Convert sparse dict to Endee sparse vector format."""
    indices = list(sparse.keys())
    values = list(sparse.values())
    return {"indices": indices, "values": values}
