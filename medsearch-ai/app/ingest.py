"""
ingest.py — Load medical articles and ingest them into Endee vector database
"""

import json
import os
import logging
from typing import List, Dict, Any
from app.embeddings import batch_embed, fit_tfidf, compute_sparse_vector
from app import endee_client

logger = logging.getLogger(__name__)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "medical_articles.json")
BATCH_SIZE = 50


def load_articles() -> List[Dict[str, Any]]:
    """Load medical articles from the JSON dataset."""
    data_path = os.path.abspath(DATA_PATH)
    logger.info(f"Loading articles from: {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        articles = json.load(f)
    logger.info(f"Loaded {len(articles)} articles.")
    return articles


def prepare_corpus(articles: List[Dict]) -> List[str]:
    """Combine title + abstract for TF-IDF training corpus."""
    return [f"{a.get('title', '')} {a.get('abstract', '')}" for a in articles]


def ingest_all(force_recreate: bool = False) -> Dict[str, Any]:
    """
    Full ingestion pipeline:
    1. Load articles
    2. Fit TF-IDF on corpus
    3. Create Endee index
    4. Embed + upsert in batches
    """
    logger.info("═" * 50)
    logger.info("Starting MedSearch AI data ingestion...")
    logger.info("═" * 50)

    # Step 1: Load articles
    articles = load_articles()
    corpus = prepare_corpus(articles)

    # Step 2: Fit TF-IDF on full corpus
    fit_tfidf(corpus)

    # Step 3: Create Endee dense index
    logger.info("Creating Endee vector index...")
    success = endee_client.create_dense_index(force_recreate=force_recreate)
    if not success:
        msg = "Failed to create Endee index. Ensure Endee is running on localhost:8080"
        logger.error(msg)
        return {"success": False, "error": msg}

    # Step 4: Embed and upsert in batches
    total_ingested = 0
    failed_batches = 0

    for batch_start in range(0, len(articles), BATCH_SIZE):
        batch = articles[batch_start: batch_start + BATCH_SIZE]
        texts = [f"{a.get('title', '')} {a.get('abstract', '')}" for a in batch]

        logger.info(f"Embedding batch {batch_start // BATCH_SIZE + 1} ({len(batch)} docs)...")
        vectors = batch_embed(texts)

        docs_to_upsert = []
        for article, vector in zip(batch, vectors):
            docs_to_upsert.append({
                "id": str(article["id"]),
                "vector": vector,
                "meta": {
                    "title": article.get("title", ""),
                    "abstract": article.get("abstract", ""),
                    "category": article.get("category", "General"),
                    "year": str(article.get("year", "")),
                    "source": article.get("source", ""),
                }
            })

        ok = endee_client.upsert_documents(docs_to_upsert)
        if ok:
            total_ingested += len(batch)
            logger.info(f"✓ Ingested {total_ingested}/{len(articles)} articles")
        else:
            failed_batches += 1
            logger.warning(f"✗ Batch at offset {batch_start} failed")

    summary = {
        "success": True,
        "total_articles": len(articles),
        "ingested": total_ingested,
        "failed_batches": failed_batches,
    }
    logger.info(f"Ingestion complete: {summary}")
    return summary
