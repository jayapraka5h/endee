"""
endee_client.py — Endee Vector Database client wrapper
Uses the official Endee Python SDK (pip install endee)
"""

import os
import logging
from typing import List, Dict, Any, Optional

from endee import Endee, Precision
from endee.schema import VectorItem  # For monkey-patching
from dotenv import load_dotenv
import httpx

load_dotenv()

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# PATCH: Fix Endee SDK Pydantic bug
# ──────────────────────────────────────────────
def _patch_vector_item_get():
    """Monkey-patch VectorItem.get() to fix SDK's Pydantic bug.
    
    The Endee SDK v0.1.19 tries to call .get() on VectorItem objects
    (which are Pydantic models, not dicts), causing AttributeError.
    
    This adds a .get() method that uses getattr() to mimic dict behavior.
    """
    def get_method(self, key, default=None):
        return getattr(self, key, default)
    
    if not hasattr(VectorItem, 'get'):
        VectorItem.get = get_method
        logger.info("VectorItem Pydantic patch applied")

_patch_vector_item_get()

ENDEE_AUTH_TOKEN = os.getenv("ENDEE_AUTH_TOKEN", None)

# The Endee SDK's base_url must include /api/v1
# Default is http://localhost:8080/api/v1
_raw_url = os.getenv("ENDEE_BASE_URL", "http://localhost:8080")
ENDEE_BASE_URL = _raw_url.rstrip("/") + "/api/v1" if not _raw_url.endswith("/api/v1") else _raw_url

DENSE_INDEX_NAME = "medsearch_dense"
EMBEDDING_DIM = 384

# Singleton Endee client
_client: Optional[Endee] = None


def _get_client() -> Endee:
    """Get or create the Endee SDK client singleton."""
    global _client
    if _client is None:
        _client = Endee(token=ENDEE_AUTH_TOKEN)
        # Direct attribute assignment bypasses set_base_url validation for local server
        _client.base_url = ENDEE_BASE_URL
        logger.info(f"Endee client initialized → {ENDEE_BASE_URL}")
    return _client


# ──────────────────────────────────────────────────
# Health Check
# ──────────────────────────────────────────────────

def check_health() -> bool:
    """Check if Endee server is running and reachable."""
    # ENDEE_BASE_URL = http://localhost:8080/api/v1
    # health endpoint = http://localhost:8080/api/v1/health
    health_url = ENDEE_BASE_URL + "/health"
    try:
        resp = httpx.get(health_url, timeout=5)
        return resp.status_code == 200
    except Exception as e:
        logger.warning(f"Endee health check failed: {e}")
        return False


def list_indexes() -> List[str]:
    """Return list of existing index names."""
    try:
        client = _get_client()
        indexes = client.list_indexes()
        names = []
        for idx in indexes:
            if isinstance(idx, dict):
                names.append(idx.get("name", idx.get("index_name", "")))
            else:
                names.append(str(idx))
        return names
    except Exception as e:
        logger.error(f"Failed to list indexes: {e}")
        return []


# ──────────────────────────────────────────────────
# Index Management
# ──────────────────────────────────────────────────

def create_dense_index(force_recreate: bool = False) -> bool:
    """Create the medsearch_dense cosine index using Endee SDK."""
    client = _get_client()

    if force_recreate:
        try:
            client.delete_index(DENSE_INDEX_NAME)
            logger.info(f"Deleted existing index '{DENSE_INDEX_NAME}'.")
        except Exception as e:
            logger.warning(f"Could not delete index (might not exist): {e}")

    try:
        client.create_index(
            name=DENSE_INDEX_NAME,
            dimension=EMBEDDING_DIM,
            space_type="cosine",
            M=16,
            ef_con=128,
            precision=Precision.INT8,
            sparse_dim=0,
        )
        logger.info(f"Dense index '{DENSE_INDEX_NAME}' created.")
        return True
    except Exception as e:
        if "already exists" in str(e).lower() or "conflict" in str(e).lower():
            logger.info(f"Index '{DENSE_INDEX_NAME}' already exists — skipping creation.")
            return True
        logger.error(f"Failed to create dense index: {e}")
        return False


# ──────────────────────────────────────────────────
# Document Upsert
# ──────────────────────────────────────────────────

def upsert_documents(documents: List[Dict[str, Any]]) -> bool:
    """
    Upsert documents to Endee using the (patched) SDK.
    The VectorItem Pydantic bug has been patched above.
    """
    try:
        client = _get_client()
        index = client.get_index(DENSE_INDEX_NAME)
        
        # Prepare documents - SDK expects list of dicts with id, vector, meta
        items = []
        for doc in documents:
            meta = doc.get("meta", {})
            item_dict = {
                "id": str(doc["id"]),
                "vector": [float(v) for v in doc["vector"]],
                "meta": {
                    "title": str(meta.get("title", "")),
                    "abstract": str(meta.get("abstract", "")),
                    "source": str(meta.get("source", "")),
                    "category": str(meta.get("category", "General")),
                    "year": str(meta.get("year", "")),
                },
            }
            items.append(item_dict)
        
        # Upsert via patched SDK
        index.upsert(items)
        logger.info(f"✓ Upserted {len(items)} documents via SDK")
        return True
            
    except Exception as e:
        logger.error(f"Upsert failed: {e}", exc_info=True)
        return False


# ──────────────────────────────────────────────────
# Vector Search
# ──────────────────────────────────────────────────

def dense_search(
    query_vector: List[float],
    top_k: int = 10,
    category_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Perform cosine similarity search with optional category payload filter.
    Returns list of {id, score, meta} dicts (sorted by descending score).
    """
    try:
        client = _get_client()
        index = client.get_index(DENSE_INDEX_NAME)

        # Fetch more results than top_k when filtering client-side
        fetch_k = top_k * 3 if category_filter else top_k

        raw = index.query(
            vector=[float(v) for v in query_vector],
            top_k=min(fetch_k, 50),
            ef=256,
        )

        results = _normalize_results(raw)

        # Client-side category filter (workaround for SDK Pydantic bug with filter)
        if category_filter and category_filter.strip().lower() not in ("", "all"):
            cat_lower = category_filter.strip().lower()
            results = [r for r in results if r["meta"].get("category", "").lower() == cat_lower]

        return results[:top_k]

    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        return []


def _normalize_results(raw: List[Dict]) -> List[Dict[str, Any]]:
    """Convert Endee raw result list to our internal {id, score, meta} format."""
    results = []
    for item in raw:
        meta = item.get("meta", {}) or {}
        results.append({
            "id": str(item.get("id", "")),
            "score": float(item.get("similarity", 0.0)),
            "meta": {
                "title": meta.get("title", ""),
                "abstract": meta.get("abstract", ""),
                "source": meta.get("source", ""),
                "category": meta.get("category", "General"),
                "year": meta.get("year", ""),
            },
        })
    return results


# ──────────────────────────────────────────────────
# Index Info
# ──────────────────────────────────────────────────

def get_index_stats() -> Dict[str, Any]:
    """Return basic stats for the medsearch_dense index."""
    try:
        client = _get_client()
        index = client.get_index(DENSE_INDEX_NAME)
        return {
            "name": index.name,
            "count": index.count,
            "dimension": index.dimension,
            "space_type": index.space_type,
            "precision": str(index.precision),
        }
    except Exception as e:
        logger.warning(f"Could not get index stats: {e}")
        return {}
