import httpx
from app.config import settings
from app.ingestion.embedder import model  # ← importer le modèle directement

COLLECTION_NAME = "documents"
BASE_URL = f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}"

# Prefix pour les questions — différent du prefix d'ingestion
# "search_query:" indique au modèle que ce texte est une QUESTION
# nomic-embed-text-v1.5 optimise l'embedding selon ce contexte
QUERY_PREFIX = "search_query: "


def retrieve(question: str, k: int = 3, score_threshold: float = 0.5) -> list[dict]:
    """
    Vectorise la question avec le prefix "search_query:"
    puis cherche les K chunks les plus proches dans Qdrant.
    """

    # Vectoriser la question avec le bon prefix
    # on utilise le même modèle que embedder.py — déjà chargé en RAM
    # on évite de le recharger (Singleton partagé)
    question_vector = model.encode(
        [QUERY_PREFIX + question],   # prefix "search_query:" + question
        convert_to_numpy=True
    )[0].tolist()                    # premier vecteur, converti en liste Python

    # Envoyer le vecteur à Qdrant pour chercher les K plus proches
    response = httpx.post(
        f"{BASE_URL}/collections/{COLLECTION_NAME}/points/search",
        json={
            "vector"      : question_vector,
            "limit"       : k,
            "with_payload": True
        },
        timeout=30
    )
    response.raise_for_status()

    results  = response.json()["result"]
    filtered = [r for r in results if r["score"] >= score_threshold]

    return filtered