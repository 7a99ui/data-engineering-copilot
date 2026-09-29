# 🤖 Data Engineering Copilot

An AI-powered assistant for Data Engineers — built with RAG, local LLM, and vector search.

## Stack
- **LLM** : Ollama + Qwen2.5:3b (local, free)
- **Embeddings** : nomic-embed-text (via Ollama)
- **API** : FastAPI + Uvicorn
- **Vector DB** : Qdrant
- **Database** : PostgreSQL
- **Search** : Dense + BM25 + RRF + Reranking

## Sprints
| Sprint | Status | Description |
|--------|--------|-------------|
| Sprint 1 — Environment & LLM | ✅ Done | Docker + Ollama + FastAPI |
| Sprint 2 — Ingestion | ✅ Done | Document pipeline + Qdrant |
| Sprint 3 — RAG | ✅ Done | Vector search + LLM generation |
| Sprint 4 — Hybrid RAG | ✅ Done | BM25 + RRF + Reranking |
| Sprint 5 — Agent | 🔜 Next | LangGraph |

## Prerequisites
- Docker Desktop
- Python 3.11+

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/data-engineering-copilot.git
cd data-engineering-copilot

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your values

# 5. Start Docker services
docker compose up -d

# 6. Pull models
docker exec copilot-ollama ollama pull qwen2.5:3b
docker exec copilot-ollama ollama pull nomic-embed-text

# 7. Ingest your documents
python -u ingest.py --source ./docs

# 8. Start the API
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Usage

- API : `http://localhost:8000`
- Swagger UI : `http://localhost:8000/docs`

### Endpoints
- `GET /health` — API status
- `POST /chat` — Direct LLM chat (no RAG)
- `POST /ask` — RAG query with sources

### Example
```json
POST /ask
{
  "question": "Why does my Spark job fail with OutOfMemoryError ?",
  "k": 3,
  "mode": "hybrid"
}
```