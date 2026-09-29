# On importe les deux pipelines — dense seul et hybride
from app.rag.retriever import retrieve
from app.rag.hybrid_retriever import hybrid_retrieve
from app.rag.reranker import rerank
from app.rag.generator import generate


async def rag_pipeline(question: str, k: int = 3, mode: str = "hybrid") -> dict:
    """
    Orchestre le pipeline RAG complet.

    question : la question de l'utilisateur
    k        : nombre de chunks finaux à donner au LLM
    mode     : stratégie de retrieval
               "dense"  → Sprint 3 (dense seul, sans reranker)
               "hybrid" → Sprint 4 (dense + BM25 + RRF + reranker)

    Retourne : {answer, sources, model, mode}
    """

    # ── Étape 1 : Retrieval selon le mode ──
    if mode == "hybrid":
        # Sprint 4 — pipeline hybride complet
        # dense + BM25 → RRF fusion → top-6 chunks
        chunks = hybrid_retrieve(question, k=k)

        # Reranker — affine les top-6 en top-k
        # lit question + chunk ensemble pour un score précis
        chunks = await rerank(question, chunks, top_k=k)
        # Filtrer les chunks avec score trop bas après reranking
        # score 0.0 = le LLM a jugé ce chunk complètement hors sujet
        chunks = [c for c in chunks if c["score"] > 0.1]

        # Si le filtre a tout éliminé → fallback dense
        if not chunks:
            chunks = retrieve(question, k=k, score_threshold=0.4)

    else:
        # Sprint 3 — dense seul (mode "dense")
        # utilisé pour comparaison ou fallback
        chunks = retrieve(question, k=k, score_threshold=0.4)

    # ── Étape 2 : Generation ──
    # même générateur qu'au Sprint 3 — inchangé
    answer = await generate(question, chunks)

    # ── Étape 3 : Construire les sources ──
    sources = []
    for chunk in chunks:
        sources.append({
            "source": chunk["payload"]["source"],
            "page"  : chunk["payload"].get("page", 1),

            # score = score du reranker (0-1) ou RRF selon le mode
            "score" : round(chunk["score"], 2),

            # extrait des 150 premiers caractères du chunk
            "chunk" : chunk["payload"]["text"][:150] + "..."
        })

    return {
        "answer" : answer,
        "sources": sources,
        "model"  : "qwen2.5:3b",

        # on retourne le mode utilisé pour que l'utilisateur sache
        # quelle stratégie a été appliquée
        "mode"   : mode
    }