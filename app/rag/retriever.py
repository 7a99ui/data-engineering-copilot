# httpx — pour faire des requêtes HTTP vers Qdrant
# même bibliothèque qu'on utilise pour Ollama — pas de nouvelle dépendance
import httpx

# settings — pour récupérer QDRANT_HOST et QDRANT_PORT depuis le .env
from app.config import settings

# embed_texts — la fonction qu'on a créée dans embedder.py au Sprint 2
# on l'importe ici car on doit vectoriser la question AVANT de chercher
# RÈGLE D'OR : même modèle pour ingestion ET pour recherche
from app.ingestion.embedder import embed_texts

# Le nom de la collection Qdrant où nos chunks sont stockés
# doit être IDENTIQUE à celui dans store.py — c'est la même "table"
COLLECTION_NAME = "documents"

# L'URL de base de Qdrant construite depuis le .env
# résultat : "http://localhost:6333"
# tous les appels HTTP vers Qdrant commencent par cette URL
BASE_URL = f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}"


def retrieve(question: str, k: int = 3, score_threshold: float = 0.4) -> list[dict]:
    """
    Vectorise la question puis cherche les K chunks
    les plus proches dans Qdrant.
    Retourne uniquement les chunks avec score > score_threshold.
    """
    # question        : la question posée par l'utilisateur en anglais
    # k               : nombre de chunks à retourner (défaut 3)
    #                   → assez de contexte sans dépasser la fenêtre du LLM
    # score_threshold : score minimum de similarité cosine accepté (défaut 0.5)
    #                   → en dessous de 0.5 = chunk trop peu pertinent
    # -> list[dict]   : retourne une liste de dicts
    #                   chaque dict contient "score" + "payload" (texte + métadonnées)

    # ── Étape 1 : Vectoriser la question ──
    # On transforme la question en vecteur de 768 nombres
    # POURQUOI : Qdrant stocke des vecteurs — pour chercher,
    #            il faut comparer un vecteur avec des vecteurs
    #
    # embed_texts([question]) → on passe une LISTE d'un seul élément
    #                           car embed_texts attend toujours une liste
    # résultat : [[0.31, -0.44, 0.72, ...]]  ← liste de listes
    #
    # [0] → on prend le premier (et unique) vecteur de la liste
    # résultat final : [0.31, -0.44, 0.72, ...]  ← un seul vecteur
    question_vector = embed_texts([question])[0]

    # ── Étape 2 : Envoyer le vecteur à Qdrant ──
    # On appelle l'API REST de Qdrant pour faire la recherche
    # endpoint : POST /collections/documents/points/search
    # Qdrant va comparer question_vector avec TOUS les vecteurs stockés
    # et retourner les K plus proches par similarité cosine
    response = httpx.post(
        # URL complète de l'endpoint de recherche Qdrant
        # ex: "http://localhost:6333/collections/documents/points/search"
        f"{BASE_URL}/collections/{COLLECTION_NAME}/points/search",

        json={
            # Le vecteur de la question — c'est ce qu'on compare
            # avec les 17 vecteurs stockés dans Qdrant
            "vector": question_vector,

            # Nombre maximum de résultats à retourner
            # si k=3 → Qdrant retourne les 3 chunks les plus proches
            # même s'il y en a 17 en tout dans la collection
            "limit": k,

            # True = inclure le payload (texte + métadonnées) dans les résultats
            # False = retourner seulement les IDs et scores (inutile pour nous)
            # INDISPENSABLE : sans payload, on ne peut pas récupérer le texte
            # à donner au LLM
            "with_payload": True
        },

        # Timeout de 30 secondes — la recherche vectorielle est rapide
        # (moins d'une seconde en général) donc 30s est très largement suffisant
        timeout=30
    )

    # Si Qdrant répond avec une erreur (500, 404...)
    # raise_for_status() lève une exception Python claire immédiatement
    # sans ça, le code continuerait avec une réponse vide ou incorrecte
    response.raise_for_status()

    # ── Étape 3 : Extraire les résultats ──
    # Qdrant retourne un JSON de cette forme :
    # {
    #   "result": [
    #     {
    #       "id"     : "f47ac10b-...",
    #       "score"  : 0.94,
    #       "payload": {
    #         "text"       : "OutOfMemoryError survient quand...",
    #         "source"     : "spark.md",
    #         "page"       : 1,
    #         "chunk_index": 2
    #       }
    #     },
    #     { ... },   ← 2ème résultat
    #     { ... }    ← 3ème résultat
    #   ]
    # }
    # On extrait uniquement la liste "result" — c'est tout ce dont on a besoin
    results = response.json()["result"]

    # ── Étape 4 : Filtrer par score minimum ──
    # On garde uniquement les chunks avec score >= score_threshold (0.5)
    # POURQUOI : un score de 0.3 signifie que le chunk est peu pertinent
    #            → le donner au LLM le confondrait plus qu'il ne l'aiderait
    #
    # C'est une list comprehension — équivalent à :
    # filtered = []
    # for r in results:
    #     if r["score"] >= score_threshold:
    #         filtered.append(r)
    #
    # Exemple concret :
    # results   = [score:0.94, score:0.87, score:0.31]
    # threshold = 0.5
    # filtered  = [score:0.94, score:0.87]  ← le 0.31 est éliminé
    filtered = [r for r in results if r["score"] >= score_threshold]

    # On retourne la liste des chunks pertinents
    # chaque élément est un dict avec "score" et "payload"
    # ce sera utilisé par generator.py pour construire le prompt
    return filtered