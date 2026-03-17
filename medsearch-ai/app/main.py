"""
main.py — FastAPI backend for MedSearch AI
Endpoints: chat (RAG), search (semantic), health, ingest, serve frontend
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load env vars before importing modules
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import endee_client, rag_engine

# ──────────────────────────────────────────────────────
# Logging Setup
# ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("medsearch")

# ──────────────────────────────────────────────────────
# App Lifespan (startup/shutdown)
# ──────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 MedSearch AI starting up...")
    # Check Endee connectivity
    if endee_client.check_health():
        logger.info("✅ Endee vector database connected at %s", os.getenv("ENDEE_BASE_URL"))
        # Create index if it doesn't exist
        if endee_client.create_dense_index():
            logger.info(f"✅ Index '{endee_client.DENSE_INDEX_NAME}' ready")
        # Check existing indexes
        indexes = endee_client.list_indexes()
        logger.info(f"Existing indexes: {indexes}")
    else:
        logger.warning("⚠️  Endee is not reachable. Please start Endee first.")
    yield
    logger.info("MedSearch AI shutting down.")


# ──────────────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────────────

app = FastAPI(
    title="MedSearch AI",
    description="Medical Knowledge RAG Assistant powered by Endee Vector Database",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ──────────────────────────────────────────────────────
# Request / Response Models
# ──────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Medical question")
    top_k: int = Field(default=8, ge=1, le=20)
    category: Optional[str] = Field(default=None, description="Filter by medical category")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=10, ge=1, le=20)
    category: Optional[str] = None


class IngestRequest(BaseModel):
    force_recreate: bool = Field(default=False, description="Delete and re-create index before ingesting")


# ──────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_frontend():
    """Serve the main chat UI."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return HTMLResponse("<h1>MedSearch AI Backend Running</h1><p>Frontend not found.</p>")


@app.get("/api/health")
async def health_check():
    """Check server and Endee database health."""
    endee_ok = endee_client.check_health()
    indexes = endee_client.list_indexes() if endee_ok else []

    # Check if medsearch index exists and get stats
    index_stats = {}
    has_data = False
    if endee_ok:
        stats = endee_client.get_index_stats()
        if stats:
            index_stats = stats
            has_data = True

    return {
        "status": "ok",
        "endee_connected": endee_ok,
        "endee_url": os.getenv("ENDEE_BASE_URL", "http://localhost:8080"),
        "available_indexes": indexes,
        "medsearch_index_ready": has_data,
        "index_stats": index_stats,
        "gemini_configured": bool(os.getenv("GEMINI_API_KEY")),
    }


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    RAG Chat endpoint — retrieves relevant medical articles from Endee,
    generates an AI answer using Gemini 1.5 Flash, returns answer + sources.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    logger.info(f"Chat request: '{request.query}' | category={request.category}")

    try:
        result = await rag_engine.rag_pipeline(
            query=request.query,
            top_k=request.top_k,
            category_filter=request.category,
        )
        return result
    except Exception as e:
        logger.error(f"Chat pipeline error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {str(e)}")


@app.post("/api/search")
async def semantic_search(request: SearchRequest):
    """
    Pure semantic search — returns ranked medical articles without LLM generation.
    Demonstrates Endee's vector similarity search with optional payload filtering.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    logger.info(f"Semantic search: '{request.query}' | category={request.category}")

    try:
        result = await rag_engine.semantic_search_only(
            query=request.query,
            top_k=request.top_k,
            category_filter=request.category,
        )
        return result
    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@app.post("/api/ingest")
async def ingest_data(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Trigger data ingestion into Endee vector database.
    Loads medical articles, generates embeddings, and stores them in Endee.
    """
    from app.ingest import ingest_all

    logger.info(f"Starting ingestion (force_recreate={request.force_recreate})...")
    try:
        result = ingest_all(force_recreate=request.force_recreate)
        return result
    except Exception as e:
        logger.error(f"Ingestion error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.get("/api/categories")
async def get_categories():
    """Return available medical categories for filtering."""
    return {
        "categories": [
            "All",
            "Cardiology",
            "Neurology",
            "Oncology",
            "Infectious Disease",
            "Endocrinology",
            "Pulmonology",
            "Gastroenterology",
            "Orthopedics",
            "Psychiatry",
            "General Medicine",
        ]
    }


@app.get("/api/sample-questions")
async def get_sample_questions():
    """Return sample medical questions to try in the UI."""
    return {
        "questions": [
            "What are the symptoms and causes of type 2 diabetes?",
            "How does hypertension affect the heart and blood vessels?",
            "What are the early warning signs of a stroke?",
            "How is breast cancer diagnosed and treated?",
            "What is the difference between bacterial and viral pneumonia?",
            "What causes Alzheimer's disease and how does it progress?",
            "How does chemotherapy work to fight cancer?",
            "What are the risk factors for coronary artery disease?",
            "How is depression treated with medication and therapy?",
            "What are the symptoms of thyroid disorders?",
        ]
    }
