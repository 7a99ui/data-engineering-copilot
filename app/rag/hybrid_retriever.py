# On importe les deux retrievers — dense et BM25
from app.rag.retriever import retrieve as dense_retrieve
from app.rag.bm25_retriever import bm25_search


def reciprocal_rank_fusion(
    dense_results : list[dict],
    bm25_results  : list[dict],
    k_rrf         : int = 60,
    top_k         : int = 6
) -> list[dict]:
    """
    Fusionne deux listes de résultats avec l'algorithme RRF.
    Retourne les top_k meilleurs chunks combinés.

    dense_results : chunks retournés par la recherche dense
    bm25_results  : chunks retournés par BM25
    k_rrf         : constante RRF (60 est la valeur standard recommandée)
                    évite de trop avantager le rang 1
    top_k         : nombre de chunks à retourner après fusion
    """

    # Dictionnaire pour accumuler les scores RRF de chaque chunk
    # clé   : le texte du chunk (identifiant unique)
    # valeur: score RRF cumulé des deux listes
    rrf_scores = {}

    # Dictionnaire pour retrouver le payload de chaque chunk
    # on en aura besoin à la fin pour construire le résultat
    chunk_payloads = {}

    # ── Traiter la liste Dense ──
    # enumerate() donne le rang (0, 1, 2...) et le chunk
    for rank, chunk in enumerate(dense_results):

        # Clé unique = texte du chunk (tronqué à 100 chars pour fiabilité)
        # on ne peut pas utiliser l'ID car BM25 n'a pas les mêmes IDs
        key = chunk["payload"]["text"][:100]

        # Formule RRF : 1 / (k_rrf + rang + 1)
        # rang + 1 car enumerate commence à 0 et le 1er rang doit être 1
        # k_rrf=60 → le rang 1 donne 1/61=0.0164, le rang 10 donne 1/70=0.0143
        # la différence entre les rangs est modérée — pas de domination totale du rang 1
        rrf_score = 1.0 / (k_rrf + rank + 1)

        # Ajouter au score total de ce chunk
        # si le chunk apparaît aussi dans BM25, son score sera additionné plus bas
        rrf_scores[key]     = rrf_scores.get(key, 0) + rrf_score
        chunk_payloads[key] = chunk["payload"]

    # ── Traiter la liste BM25 ──
    # même logique mais pour BM25
    for rank, chunk in enumerate(bm25_results):
        key = chunk["payload"]["text"][:100]

        rrf_score = 1.0 / (k_rrf + rank + 1)

        # Si ce chunk était déjà dans Dense → son score s'additionne
        # c'est l'avantage clé de RRF : un chunk bien classé dans LES DEUX
        # listes obtient un score bien supérieur
        rrf_scores[key]     = rrf_scores.get(key, 0) + rrf_score
        chunk_payloads[key] = chunk["payload"]

    # ── Trier par score RRF décroissant ──
    # items() retourne des paires (clé, valeur)
    # on trie par valeur (score RRF) en ordre décroissant
    sorted_chunks = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # ── Construire le résultat final ──
    # on garde les top_k meilleurs et on retourne dans le format standard
    results = []
    for key, score in sorted_chunks[:top_k]:
        results.append({
            # score RRF — pas comparable à cosine mais utilisé pour le tri
            "score"  : round(score, 4),
            "payload": chunk_payloads[key]
        })

    return results


def hybrid_retrieve(question: str, k: int = 3) -> list[dict]:
    """
    Pipeline de retrieval hybride complet :
    1. Dense search  → top-10 par similarité vectorielle
    2. BM25 search   → top-10 par correspondance lexicale
    3. RRF fusion    → top-6 meilleurs combinés
    Retourne les k meilleurs chunks fusionnés.
    """

    # ── Étape 1 : Dense search ──
    # on récupère plus de résultats qu'au Sprint 3 (10 au lieu de 3)
    # car on va affiner avec RRF et le reranker ensuite
    # score_threshold=0.3 → plus permissif pour laisser BM25 compenser
    dense_results = dense_retrieve(
        question,
        k=10,
        score_threshold=0.3
    )

    # ── Étape 2 : BM25 search ──
    # cherche les chunks qui contiennent les mots de la question
    bm25_results = bm25_search(question, k=10)

    # ── Étape 3 : RRF fusion ──
    # fusionne les deux listes et retourne les top-6
    # top_k=6 → on garde 6 chunks pour le reranker
    # (le reranker affine ensuite à k=3)
    fused_results = reciprocal_rank_fusion(
        dense_results,
        bm25_results,
        top_k=6
    )

    # Si la fusion n'a rien retourné → fallback sur dense seul
    if not fused_results:
        return dense_retrieve(question, k=k)

    return fused_results