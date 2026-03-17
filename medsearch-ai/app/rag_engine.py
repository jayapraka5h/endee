"""
rag_engine.py — RAG pipeline: embed query → retrieve from Endee → generate with Gemini
"""

import os
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
import google.generativeai as genai
from dotenv import load_dotenv

from app.embeddings import embed_text, compute_sparse_vector
from app import endee_client

load_dotenv()

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-1.5-flash"

# Configure Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


# ──────────────────────────────────────────────────────
# RAG System Prompt
# ──────────────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """You are MedSearch AI, an expert medical information assistant powered by a semantic vector database.

Your role is to provide accurate, helpful medical information based ONLY on the retrieved medical articles provided to you.

Guidelines:
- Answer clearly and concisely using information from the provided context
- Always cite the source articles you use (mention their titles)
- If the context doesn't contain relevant information, say so honestly
- Do NOT make up information not present in the context
- Recommend consulting a healthcare professional for personal medical decisions
- Use plain language, avoid excessive medical jargon unless explaining the term
- Structure longer answers with bullet points or numbered lists for clarity

Format your response as:
1. A direct answer to the question
2. Key details/explanations with bullet points
3. A brief note citing which articles were used
4. A disclaimer to consult a doctor for personal health decisions (keep brief)
"""


# ──────────────────────────────────────────────────────
# Core RAG Functions
# ──────────────────────────────────────────────────────

def retrieve_documents(
    query: str,
    top_k: int = 8,
    category_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Embed query and retrieve top-k relevant medical documents from Endee.
    Uses dense cosine similarity search with optional payload filtering.
    """
    # 1. Embed the query to dense vector
    query_vector = embed_text(query)

    # 2. Search Endee with optional category filter
    results = endee_client.dense_search(
        query_vector=query_vector,
        top_k=top_k,
        category_filter=category_filter,
    )

    logger.info(f"Retrieved {len(results)} documents for query: '{query[:60]}...'")
    return results


def build_context(results: List[Dict[str, Any]]) -> str:
    """Format retrieved documents into a context string for the LLM."""
    if not results:
        return "No relevant medical articles found."

    context_parts = []
    for i, result in enumerate(results, 1):
        meta = result.get("meta", {})
        title = meta.get("title", "Unknown Title")
        abstract = meta.get("abstract", "No content available.")
        category = meta.get("category", "General")
        year = meta.get("year", "")
        score = result.get("score", 0)

        context_parts.append(
            f"[Article {i}] (Relevance: {score:.2f}, Category: {category}, Year: {year})\n"
            f"Title: {title}\n"
            f"Content: {abstract}\n"
        )

    return "\n---\n".join(context_parts)


def generate_answer_stream(query: str, context: str) -> str:
    """
    Generate an answer using Gemini with RAG context.
    Returns the full generated text.
    """
    if not GEMINI_API_KEY:
        return _fallback_answer(query, context)

    try:
        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=RAG_SYSTEM_PROMPT,
        )

        prompt = f"""Based on the following medical articles, answer the user's question.

RETRIEVED MEDICAL ARTICLES:
{context}

USER QUESTION: {query}

Please provide a comprehensive, accurate answer based solely on the context above."""

        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.3,
                max_output_tokens=1024,
            )
        )
        return response.text

    except Exception as e:
        logger.error(f"Gemini generation failed: {e}")
        return _fallback_answer(query, context)


async def rag_pipeline(
    query: str,
    top_k: int = 8,
    category_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Full RAG pipeline: retrieve → generate → return answer + sources.
    """
    # Step 1: Retrieve relevant documents
    results = retrieve_documents(query, top_k=top_k, category_filter=category_filter)

    # Step 2: Build context from retrieved docs
    context = build_context(results)

    # Step 3: Generate answer with Gemini
    answer = generate_answer_stream(query, context)

    # Step 4: Format sources for frontend
    sources = []
    for r in results:
        meta = r.get("meta", {})
        sources.append({
            "id": r.get("id", ""),
            "title": meta.get("title", "Unknown"),
            "category": meta.get("category", "General"),
            "year": meta.get("year", ""),
            "source": meta.get("source", ""),
            "abstract_snippet": meta.get("abstract", "")[:200] + "...",
            "relevance_score": round(r.get("score", 0), 3),
        })

    return {
        "answer": answer,
        "sources": sources,
        "query": query,
        "total_retrieved": len(results),
    }


async def semantic_search_only(
    query: str,
    top_k: int = 10,
    category_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """Pure semantic search without LLM generation — returns ranked documents."""
    results = retrieve_documents(query, top_k=top_k, category_filter=category_filter)

    sources = []
    for r in results:
        meta = r.get("meta", {})
        sources.append({
            "id": r.get("id", ""),
            "title": meta.get("title", "Unknown"),
            "category": meta.get("category", "General"),
            "year": meta.get("year", ""),
            "source": meta.get("source", ""),
            "abstract": meta.get("abstract", ""),
            "relevance_score": round(r.get("score", 0), 3),
        })

    return {
        "query": query,
        "results": sources,
        "total": len(sources),
    }


def _fallback_answer(query: str, context: str) -> str:
    """Fallback when Gemini API is unavailable — summarize retrieved context."""
    if "No relevant" in context:
        return (
            f"I couldn't find specific information about '{query}' in the medical database. "
            "Please try rephrasing your query or consult a healthcare professional."
        )

    lines = context.split("\n")
    titles = [l.replace("Title: ", "").strip() for l in lines if l.startswith("Title:")]
    title_list = "\n".join(f"• {t}" for t in titles[:3])

    return (
        f"Based on the medical knowledge base, here are the most relevant articles for '{query}':\n\n"
        f"{title_list}\n\n"
        "Please review the source articles in the panel on the right for detailed information. "
        "Always consult a qualified healthcare professional for personalized medical advice."
    )
