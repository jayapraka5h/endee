<p align="center">
  <img src="https://img.shields.io/badge/Vector_DB-Endee-6366f1?style=for-the-badge" alt="Endee" />
  <img src="https://img.shields.io/badge/LLM-Gemini_1.5_Flash-06b6d4?style=for-the-badge" alt="Gemini" />
  <img src="https://img.shields.io/badge/Framework-FastAPI-10b981?style=for-the-badge" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Embeddings-sentence--transformers-f59e0b?style=for-the-badge" alt="Embeddings" />
  <img src="https://img.shields.io/badge/License-Apache_2.0-gray?style=for-the-badge" alt="License" />
</p>

<h1 align="center">🧬 MedSearch AI</h1>
<p align="center"><b>Medical Knowledge RAG Assistant powered by Endee Vector Database</b></p>

---

## 📖 Project Overview

**MedSearch AI** is a full-stack AI/ML application that demonstrates a **production-quality RAG (Retrieval-Augmented Generation)** pipeline using [Endee](https://github.com/endee-io/endee) as the high-performance vector database backend.

Users can ask natural-language **medical questions** and receive AI-generated answers with cited source documents — all retrieved in milliseconds from Endee's vector index.

```
User Query → Embed with sentence-transformers → Search Endee (dense vector + payload filter)
           → Retrieved articles as context → Gemini 1.5 Flash → Cited answer + sources
```

---

## 🏗️ System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                        MedSearch AI System                       │
├────────────────┬────────────────────────┬────────────────────────┤
│   Frontend     │     FastAPI Backend     │   Endee Vector DB      │
│  (HTML/CSS/JS) │    (Python 3.11)        │   (Port 8080)          │
│                │                        │                        │
│  Chat UI       │  /api/chat  (RAG)       │  Dense Index           │
│  Sources Panel │  /api/search (Semantic) │  - 384-dim cosine      │
│  Mode Toggle   │  /api/ingest            │  - INT8 precision      │
│  Category      │  /api/health            │  - HNSW (M=16)         │
│  Filter        │                        │  - Payload filter      │
│                │  sentence-transformers  │    (category, year)    │
│                │  all-MiniLM-L6-v2       │                        │
│                │  Google Gemini 1.5 Flash│  50 Medical Articles   │
│                │  TF-IDF sparse vecs     │  10 Specialties        │
└────────────────┴────────────────────────┴────────────────────────┘
```

---

## 🚀 Why Endee?

| Feature | How MedSearch AI Uses It |
|---|---|
| **Dense vector search** | Cosine similarity with 384-dim sentence embeddings for semantic article retrieval |
| **Payload filtering** | Filter results by medical category (Cardiology, Neurology, etc.) using Endee's metadata filter |
| **INT8 precision** | Memory-efficient storage of vectors without quality loss |
| **HNSW indexing** | Sub-millisecond nearest-neighbor search across 50+ medical documents |
| **HTTP REST API** | Python SDK (`pip install endee`) used for index creation, upsert, and query |
| **Docker deployment** | Endee runs as a standalone Docker container alongside the FastAPI backend |

---

## 🎯 Features

- 🤖 **RAG Chat Mode** — Ask questions and get AI-generated answers with cited sources
- 🔍 **Semantic Search Mode** — Pure vector similarity search returning ranked articles
- 🏷️ **Category Filtering** — Payload-filtered search by medical specialty (Endee feature)
- 🎛️ **Top-K Control** — Adjust how many documents Endee retrieves (3-15)
- 📚 **Sources Panel** — Shows retrieved Endee documents with relevance scores
- ⚡ **Health Monitoring** — Real-time Endee connection status in the header
- 🌑 **Glassmorphism Dark UI** — Premium, responsive dark-mode interface

---

## 📁 Project Structure

```
medsearch-ai/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI backend (chat, search, ingest endpoints)
│   ├── rag_engine.py    # RAG pipeline (embed → retrieve → generate)
│   ├── endee_client.py  # Endee HTTP API wrapper (index, upsert, query)
│   ├── embeddings.py    # sentence-transformers + TF-IDF sparse vectors
│   └── ingest.py        # Data loading and Endee ingestion pipeline
├── data/
│   └── medical_articles.json  # 50 medical articles, 10 specialties
├── frontend/
│   └── index.html       # Full chat UI (HTML + CSS + JS)
├── scripts/
│   └── run_ingest.py    # One-shot ingestion CLI script
├── Dockerfile
├── requirements.txt
└── .env.example
docker-compose-app.yml   # Full-stack Docker Compose
```

---

## ⚙️ Setup Instructions

### Prerequisites

- Python 3.10+ (for local) **or** Docker (for containerized)
- Gemini API key (free at [https://aistudio.google.com](https://aistudio.google.com/app/apikey))

---

### Option A — Local Setup (Recommended for Development)

**Step 1: Start Endee vector database**

```bash
docker run \
  --ulimit nofile=100000:100000 \
  -p 8080:8080 \
  -v ./endee-data:/data \
  --name endee-server \
  --restart unless-stopped \
  endeeio/endee-server:latest
```

Verify: visit [http://localhost:8080](http://localhost:8080) ✅

**Step 2: Clone and set up the Python environment**

```bash
cd d:\Endee.io\medsearch-ai
pip install -r requirements.txt
```

**Step 3: Configure environment**

```bash
# Copy the example env file
copy .env.example .env

# Edit .env and add your Gemini API key:
# GEMINI_API_KEY=your_key_here
```

**Step 4: Ingest medical articles into Endee**

```bash
python scripts/run_ingest.py
```

This will:
- Load 50 medical articles from `data/medical_articles.json`
- Generate 384-dim sentence embeddings using `all-MiniLM-L6-v2`
- Create a cosine similarity index in Endee (`medsearch_dense`)
- Upsert all articles with metadata (title, abstract, category, year)

**Step 5: Start the FastAPI backend**

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Step 6: Open the app**

🌐 [http://localhost:8000](http://localhost:8000)

---

### Option B — Docker Compose (Full Stack)

```bash
# From the Endee.io root directory:
# 1. Create .env with your Gemini key
echo "GEMINI_API_KEY=your_key_here" > .env

# 2. Start both Endee and MedSearch API
docker compose -f docker-compose-app.yml up -d

# 3. Run ingestion inside the API container
docker exec medsearch-api python scripts/run_ingest.py

# 4. Open the app
# http://localhost:8000
```

---

## 🧪 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Check server + Endee connectivity |
| `POST` | `/api/chat` | RAG pipeline: retrieve + generate answer |
| `POST` | `/api/search` | Pure Endee semantic search |
| `POST` | `/api/ingest` | Trigger data ingestion into Endee |
| `GET` | `/api/categories` | List available filter categories |
| `GET` | `/api/sample-questions` | Get example medical questions |

**Chat request example:**
```json
POST /api/chat
{
  "query": "What causes Type 2 Diabetes?",
  "top_k": 8,
  "category": "Endocrinology"
}
```

**Chat response:**
```json
{
  "answer": "Type 2 diabetes is caused by...",
  "sources": [
    {
      "id": "1",
      "title": "Type 2 Diabetes: Pathophysiology...",
      "category": "Endocrinology",
      "relevance_score": 0.921
    }
  ],
  "total_retrieved": 8
}
```

---

## 🗂️ Dataset

The `data/medical_articles.json` contains **50 curated medical articles** across 10 specialties:

| Specialty | Articles |
|---|---|
| Cardiology | Hypertension, AMI, Heart Failure, CAD, AF, PAD, VTE |
| Neurology | Stroke, Alzheimer's, Parkinson's, MS, Epilepsy, Migraine |
| Oncology | Breast, Colorectal, Lung, Prostate, Ovarian, HCC, Bladder, Endometrial |
| Infectious Disease | COVID-19, Pneumonia, Sepsis, TB, HIV |
| Endocrinology | Diabetes T2, Hypothyroidism, Obesity, Vitamin D, CKD |
| Pulmonology | Asthma, COPD |
| Gastroenterology | IBD, GERD, NAFLD, Acute Pancreatitis, Liver Cirrhosis |
| Psychiatry | MDD, Anxiety, Schizophrenia, Bipolar |
| Orthopedics | Osteoporosis, Kidney Stones, IDA |
| General Medicine | RA, SLE, Psoriasis, Anaphylaxis |

---

## 🧠 How Endee Is Used

```python
from endee import Endee, Precision

# Create the vector index
client = Endee()
client.create_index(
    name="medsearch_dense",
    dimension=384,
    space_type="cosine",
    precision=Precision.INT8
)

# Search with payload filter
index = client.get_index("medsearch_dense")
results = index.query(
    vector=query_embedding,  # 384-dim from sentence-transformers
    top_k=8,
    filter={"category": "Cardiology"}  # Endee payload filter
)
```

---

## 🤝 Contributing

This project was built as a demonstration of Endee's capabilities for AI retrieval workloads. Contributions welcome!

---

## 📝 License

Apache License 2.0 — see [LICENSE](../LICENSE)
