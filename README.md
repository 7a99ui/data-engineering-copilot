# 🤖 Data Engineering Copilot

Assistant IA local pour les ingénieurs data — basé sur RAG, LangGraph et LLM local.

## Stack
- **LLM** : Ollama + Qwen2.5:3b (local, gratuit)
- **API** : FastAPI + Uvicorn
- **Vector DB** : Qdrant
- **Base de données** : PostgreSQL

## Prérequis
- Docker Desktop
- Python 3.11+

## Installation

```bash
# 1. Cloner le repo
git clone https://github.com/TON_USERNAME/data-engineering-copilot.git
cd data-engineering-copilot

# 2. Créer l'environnement Python
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer l'environnement
cp .env.example .env
# Éditer .env avec tes valeurs

# 5. Lancer les services
docker compose up -d

# 6. Télécharger le modèle
docker exec copilot-ollama ollama pull qwen2.5:3b

# 7. Lancer l'API
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Utilisation

API disponible sur `http://localhost:8000`
Documentation interactive : `http://localhost:8000/docs`

## Sprints

| Sprint | Statut | Description |
|--------|--------|-------------|
| Sprint 1 — Environnement & LLM | ✅ Terminé | Docker + Ollama + FastAPI |
| Sprint 2 — Ingestion | 🔜 En cours | Pipeline de documents |
| Sprint 3 — RAG | ⏳ À venir | Recherche vectorielle |
| Sprint 4 — RAG Hybride | ⏳ À venir | BM25 + Reranking |
| Sprint 5 — Agent | ⏳ À venir | LangGraph |