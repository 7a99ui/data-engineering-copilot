# httpx pour appeler Ollama
import httpx

# settings pour l'URL d'Ollama
from app.config import settings


async def rerank(question: str, chunks: list[dict], top_k: int = 3) -> list[dict]:
    """
    Reranker basé sur Ollama — évalue la pertinence de chaque chunk
    par rapport à la question en les lisant ENSEMBLE.

    Approche : on demande au LLM de scorer chaque paire (question, chunk)
    de 0 à 10. Plus robuste que sentence-transformers sur Windows.

    question : la question de l'utilisateur
    chunks   : les chunks fusionnés par RRF (top-6)
    top_k    : nombre de chunks à retourner après reranking (défaut 3)
    -> list[dict] : les top_k chunks reordonnés par pertinence réelle
    """

    # Si pas assez de chunks → retourner directement sans reranker
    # pas besoin de reranker si on a déjà peu de chunks
    if len(chunks) <= top_k:
        return chunks

    # Liste pour accumuler les scores de pertinence
    scored_chunks = []

    for chunk in chunks:
        # Extraire le texte du chunk
        text = chunk["payload"]["text"]

        # ── Construire le prompt de scoring ──
        # On demande au LLM d'évaluer la pertinence de 0 à 10
        # Le LLM lit la question ET le chunk ensemble → CrossEncoder-like
        scoring_prompt = f"""Rate the relevance of the following passage to answer the question.
Respond with ONLY a number from 0 to 10.
0 = completely irrelevant
5 = somewhat relevant
10 = perfectly answers the question

Question: {question}

Passage: {text[:500]}

Relevance score (0-10):"""

        # ── Appeler Ollama pour scorer ce chunk ──
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{settings.OLLAMA_BASE_URL}/api/generate",
                json={
                    "model" : settings.LLM_MODEL,
                    "prompt": scoring_prompt,
                    "stream": False,
                    "options": {
                        # temperature=0 → réponse totalement déterministe
                        # on veut toujours le même score pour le même chunk
                        "temperature": 0.0,

                        # num_predict=3 → on attend juste "7" ou "10"
                        # pas besoin d'une longue réponse
                        "num_predict": 3
                    }
                }
            )

        # Extraire le texte de la réponse
        raw_score = response.json()["response"].strip()

        # ── Parser le score ──
        # Le LLM peut répondre "8", "8/10", "8." etc.
        # on extrait juste le premier nombre trouvé
        try:
            # Prendre uniquement les chiffres de la réponse
            # ex: "8/10" → "8", "7." → "7"
            score_str = ''.join(filter(str.isdigit, raw_score.split()[0]))
            score = float(score_str) if score_str else 5.0

            # S'assurer que le score est entre 0 et 10
            score = max(0.0, min(10.0, score))

        except (ValueError, IndexError):
            # Si le LLM ne retourne pas un nombre → score neutre de 5
            score = 5.0

        scored_chunks.append({
            # Normaliser le score entre 0 et 1 (diviser par 10)
            # pour cohérence avec les autres scores du système
            "score"  : score / 10.0,
            "payload": chunk["payload"],

            # Garder aussi le score RRF original pour debug
            "rrf_score": chunk["score"]
        })

    # ── Trier par score de pertinence décroissant ──
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)

    # Retourner seulement les top_k meilleurs
    return scored_chunks[:top_k]