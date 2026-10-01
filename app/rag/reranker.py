# CrossEncoder de sentence-transformers
# c'est le vrai modèle de reranking — lit question + chunk ENSEMBLE
# contrairement au BiEncoder qui les encode séparément
from sentence_transformers import CrossEncoder

# Le modèle standard de reranking en production RAG
# entraîné sur MS MARCO — dataset de 500k paires question/passage
# taille : ~86MB — beaucoup plus léger que les LLMs
# téléchargé automatiquement au premier appel depuis HuggingFace
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Chargement du modèle UNE SEULE FOIS au démarrage — pattern Singleton
# comme ollama_client et le modèle d'embeddings
# évite de recharger 86MB à chaque requête
cross_encoder = CrossEncoder(MODEL_NAME)


async def rerank(question: str, chunks: list[dict], top_k: int = 3) -> list[dict]:
    """
    Reranke les chunks avec le vrai modèle CrossEncoder.
    Évalue chaque paire (question, chunk) ENSEMBLE dans le même modèle.
    Beaucoup plus précis que le scoring via LLM.

    question : la question de l'utilisateur
    chunks   : les chunks fusionnés par RRF (top-6)
    top_k    : nombre de chunks à retourner après reranking (défaut 3)
    -> list[dict] : les top_k chunks reordonnés par pertinence réelle
    """

    # Si pas assez de chunks → retourner directement sans reranker
    if len(chunks) <= top_k:
        return chunks

    # ── Construire les paires (question, chunk) ──
    # CrossEncoder attend une liste de tuples (texte1, texte2)
    # il lira les deux ENSEMBLE dans le même passage du modèle
    # c'est fondamentalement différent du BiEncoder qui les encode séparément
    pairs = [
        (question, chunk["payload"]["text"])
        for chunk in chunks
    ]

    # ── Calculer les scores CrossEncoder ──
    # model.predict() retourne un tableau numpy de scores
    # un score par paire — pas de limite de valeur (peut être négatif)
    # score élevé = chunk très pertinent pour cette question
    # score négatif = chunk peu ou pas pertinent
    # ex: [2.4, -1.2, 0.8, -3.5, 1.1, -0.3]
    scores = cross_encoder.predict(pairs)

    # ── Associer chaque chunk à son score ──
    # zip() associe chunk[0] avec scores[0], chunk[1] avec scores[1]...
    # float() convertit le score numpy en float Python standard
    scored_chunks = [
        {
            # score brut du CrossEncoder — peut être négatif
            # on le garde tel quel pour le tri
            "raw_score": float(scores[i]),

            # score normalisé entre 0 et 1 avec sigmoid
            # sigmoid(x) = 1 / (1 + e^(-x))
            # transforme n'importe quel score en valeur 0-1
            # score 0   → sigmoid = 0.50
            # score 2   → sigmoid = 0.88
            # score -2  → sigmoid = 0.12
            # cohérent avec les autres scores du système
            "score"    : float(1 / (1 + 2.718 ** (-scores[i]))),

            # conserver le payload complet (texte + métadonnées)
            "payload"  : chunks[i]["payload"],

            # conserver le score RRF original pour traçabilité
            "rrf_score": chunks[i]["score"]
        }
        for i in range(len(chunks))
    ]

    # ── Trier par score décroissant ──
    # le chunk le plus pertinent pour la question sera en premier
    scored_chunks.sort(key=lambda x: x["raw_score"], reverse=True)

    # ── Retourner les top_k meilleurs ──
    return scored_chunks[:top_k]