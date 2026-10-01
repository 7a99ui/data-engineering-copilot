from sentence_transformers import SentenceTransformer

# nomic-embed-text-v1.5 — version améliorée du modèle utilisé avant
# même 768 dimensions — compatible avec la collection Qdrant existante
# exécuté localement en RAM — pas d'appel HTTP à Ollama
# trust_remote_code=True requis — le modèle a du code personnalisé sur HuggingFace
MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"

# Chargement une seule fois au démarrage — pattern Singleton
# ~270MB en RAM — chargé une seule fois pour toute la durée de vie de l'app
model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)

# Prefix pour l'ingestion — spécificité de nomic-embed-text-v1.5
# "search_document:" indique au modèle que ce texte sera stocké et cherché
# améliore la qualité des embeddings pour la recherche documentaire
DOCUMENT_PREFIX = "search_document: "


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Génère les embeddings localement via sentence-transformers.
    Traite tous les textes en batch — beaucoup plus rapide
    qu'un appel HTTP par texte comme avant avec Ollama.

    texts : liste de textes à vectoriser (chunks de documents)
    -> list : liste de vecteurs de 768 dimensions
    """

    # Ajouter le prefix "search_document:" à chaque chunk
    # POURQUOI : nomic-embed-text-v1.5 a été entraîné avec ces prefixes
    # "search_document:" → dit au modèle "ce texte sera cherché plus tard"
    # sans prefix → embeddings moins précis pour la recherche
    prefixed_texts = [DOCUMENT_PREFIX + text for text in texts]

    # encode() traite tout le batch en parallèle en RAM
    # AVANTAGE vs Ollama HTTP : pas de latence réseau, traitement parallèle
    # show_progress_bar=True → barre de progression visible pendant l'ingestion
    # convert_to_numpy=True → tableau numpy, plus facile à manipuler
    embeddings = model.encode(
        prefixed_texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # .tolist() convertit numpy array → liste Python standard
    # Qdrant attend une liste Python, pas un tableau numpy
    return embeddings.tolist()