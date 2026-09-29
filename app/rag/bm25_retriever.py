# rank_bm25 — bibliothèque Python pure pour BM25
# pas de dépendance PyTorch ou gRPC — fonctionne partout
from rank_bm25 import BM25Okapi

# httpx pour récupérer les chunks stockés dans Qdrant
import httpx

# settings pour l'URL de Qdrant
from app.config import settings

# re pour le nettoyage du texte (expressions régulières)
import re

# La collection Qdrant où nos chunks sont stockés
COLLECTION_NAME = "documents"

# URL de base de Qdrant
BASE_URL = f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}"


def fetch_all_chunks() -> list[dict]:
    """
    Récupère TOUS les chunks stockés dans Qdrant.
    BM25 a besoin de voir tous les documents pour calculer
    l'IDF (fréquence inverse dans le corpus complet).
    """

    # On demande à Qdrant de nous donner tous les points
    # endpoint : POST /collections/documents/points/scroll
    # "scroll" = pagination — permet de récupérer tous les points
    # même s'il y en a des milliers
    response = httpx.post(
        f"{BASE_URL}/collections/{COLLECTION_NAME}/points/scroll",
        json={
            # limit=1000 → on récupère jusqu'à 1000 chunks en une fois
            # suffisant pour notre projet (on en a 12)
            "limit"       : 1000,

            # True = inclure le texte et les métadonnées dans la réponse
            # sans ça on n'aurait que les IDs et vecteurs — inutile pour BM25
            "with_payload": True,

            # False = ne pas inclure les vecteurs (768 nombres par chunk)
            # on n'en a pas besoin pour BM25 — économise de la mémoire
            "with_vectors": False
        },
        timeout=30
    )
    response.raise_for_status()

    # Qdrant retourne {"result": {"points": [...], "next_page_offset": null}}
    # on extrait uniquement la liste des points
    return response.json()["result"]["points"]


def tokenize(text: str) -> list[str]:
    """
    Découpe un texte en liste de mots (tokens) pour BM25.
    BM25 travaille sur des mots, pas sur des vecteurs.

    Exemple :
    "Spark OutOfMemoryError — increase executor memory"
    → ["spark", "outofmemoryerror", "increase", "executor", "memory"]
    """

    # Mettre en minuscules — "Spark" et "spark" doivent être le même mot
    text = text.lower()

    # Garder uniquement les lettres et chiffres, remplacer le reste par un espace
    # re.sub remplace tous les caractères non-alphanumériques par " "
    # ex: "spark.executor.memory=4g" → "spark executor memory 4g"
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # Découper en mots et filtrer les mots trop courts (1-2 lettres)
    # "a", "to", "is" sont des mots vides — peu utiles pour BM25
    tokens = [word for word in text.split() if len(word) > 2]

    return tokens


def bm25_search(question: str, k: int = 10) -> list[dict]:
    """
    Recherche BM25 — trouve les chunks qui contiennent
    les mots de la question avec le score BM25 le plus élevé.

    question : la question de l'utilisateur
    k        : nombre de résultats à retourner (défaut 10)
    -> list[dict] : chunks avec score BM25 et payload
    """

    # ── Étape 1 : Récupérer tous les chunks de Qdrant ──
    # BM25 doit voir TOUT le corpus pour calculer les scores IDF
    # IDF = Inverse Document Frequency = à quel point un mot est rare
    # un mot rare dans le corpus a plus de valeur qu'un mot commun
    all_chunks = fetch_all_chunks()

    # Si Qdrant est vide → retourner liste vide
    if not all_chunks:
        return []

    # ── Étape 2 : Tokeniser tous les chunks ──
    # BM25Okapi attend une liste de listes de tokens
    # ex: [["spark", "executor", "memory"], ["airflow", "dag", ...], ...]
    # on extrait le texte de chaque chunk et on le tokenise
    corpus_tokens = [
        tokenize(chunk["payload"]["text"])
        for chunk in all_chunks
    ]

    # ── Étape 3 : Construire l'index BM25 ──
    # BM25Okapi analyse le corpus et calcule les statistiques IDF
    # c'est comme construire un index inversé (mot → chunks qui le contiennent)
    # doit être reconstruit à chaque appel car le corpus peut changer
    # (en production on le mettrait en cache)
    bm25_index = BM25Okapi(corpus_tokens)

    # ── Étape 4 : Tokeniser la question ──
    # La question doit être tokenisée de la même façon que les chunks
    # pour que les mots correspondent
    question_tokens = tokenize(question)

    # ── Étape 5 : Calculer les scores BM25 ──
    # get_scores() retourne un tableau numpy de scores
    # un score par chunk dans le même ordre que all_chunks
    # ex: [8.4, 0.0, 2.1, 5.3, 0.0, ...]
    scores = bm25_index.get_scores(question_tokens)

    # ── Étape 6 : Trier par score et prendre les K meilleurs ──
    # on crée des paires (index, score) et on trie par score décroissant
    scored_chunks = [
        (i, float(scores[i]))
        for i in range(len(all_chunks))
        if scores[i] > 0  # ignorer les chunks avec score 0 (aucun mot en commun)
    ]

    # sorted() trie par score décroissant (reverse=True)
    # key=lambda x: x[1] → trier sur le score (deuxième élément du tuple)
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    # Garder seulement les K meilleurs
    top_chunks = scored_chunks[:k]

    # ── Étape 7 : Construire le résultat dans le même format que retriever.py ──
    # on retourne le même format que la recherche dense pour que
    # hybrid_retriever.py puisse traiter les deux de la même façon
    results = []
    for idx, score in top_chunks:
        results.append({
            # score BM25 normalisé entre 0 et 1
            # on divise par le score max pour avoir une échelle comparable
            "score"  : score / max(s for _, s in scored_chunks),
            "payload": all_chunks[idx]["payload"],
            "id"     : all_chunks[idx]["id"]
        })

    return results