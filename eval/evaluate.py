# os pour configurer les variables d'environnement
import os

# sys pour ajouter le dossier parent au path Python
# nécessaire pour importer app.rag depuis le dossier eval/
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# json pour sauvegarder les résultats
import json

# asyncio pour exécuter les fonctions async de notre pipeline RAG
import asyncio

# Notre dataset de questions/réponses de référence
from eval.dataset import EVALUATION_DATASET

# Notre pipeline RAG — on va l'appeler directement
# sans passer par FastAPI ni par l'agent
from app.rag.retriever import retrieve
from app.rag.hybrid_retriever import hybrid_retrieve
from app.rag.reranker import rerank
from app.rag.generator import generate


# ══════════════════════════════════════════════════════
# FONCTIONS DE RETRIEVAL — les 3 stratégies à comparer
# ══════════════════════════════════════════════════════

async def run_dense(question: str) -> dict:
    """
    Stratégie 1 — Dense seul (Sprint 3).
    Vectorise la question et cherche dans Qdrant par similarité cosine.
    Pas de BM25, pas de reranking.

    question : la question à traiter
    -> dict  : {answer, contexts} pour RAGAS
    """

    # Récupérer les top-3 chunks par similarité cosine
    # retrieve() utilise le modèle nomic-embed-text-v1.5
    chunks = retrieve(question, k=3)

    # Générer la réponse avec le LLM
    # generate() construit le prompt RAG + appelle Ollama
    answer = await generate(question, chunks)

    # Extraire le texte de chaque chunk pour RAGAS
    # RAGAS a besoin du texte brut — pas de métadonnées
    contexts = [chunk["payload"]["text"] for chunk in chunks]

    return {
        "answer"  : answer,    # la réponse générée par le LLM
        "contexts": contexts   # les chunks utilisés comme contexte
    }


async def run_hybrid(question: str) -> dict:
    """
    Stratégie 2 — Hybride Dense + BM25 + RRF (Sprint 4 partiel).
    Combine deux approches de recherche puis fusionne avec RRF.
    Pas de reranking CrossEncoder.

    question : la question à traiter
    -> dict  : {answer, contexts} pour RAGAS
    """

    # Récupérer les top-6 chunks avec fusion RRF
    # hybrid_retrieve() combine Dense + BM25 + Reciprocal Rank Fusion
    chunks = hybrid_retrieve(question, k=6)

    # Prendre les top-3 sans reranker
    # on coupe à 3 pour avoir la même taille que la stratégie dense
    chunks = chunks[:3]

    # Générer la réponse
    answer = await generate(question, chunks)

    contexts = [chunk["payload"]["text"] for chunk in chunks]

    return {
        "answer"  : answer,
        "contexts": contexts
    }


async def run_hybrid_reranker(question: str) -> dict:
    """
    Stratégie 3 — Hybride + CrossEncoder Reranker (Sprint 4 final).
    Notre meilleure stratégie — celle qu'on pense être la meilleure.
    C'est ici que RAGAS devrait donner les meilleurs scores.

    question : la question à traiter
    -> dict  : {answer, contexts} pour RAGAS
    """

    # Récupérer les top-6 chunks avec fusion RRF
    chunks = hybrid_retrieve(question, k=6)

    # Reranker avec CrossEncoder ms-marco-MiniLM-L-6-v2
    # rerank() calcule un score précis pour chaque paire (question, chunk)
    reranked_chunks = await rerank(question, chunks, top_k=3)

    # Générer la réponse avec les chunks rerankés
    answer = await generate(question, reranked_chunks)

    contexts = [chunk["payload"]["text"] for chunk in reranked_chunks]

    return {
        "answer"  : answer,
        "contexts": contexts
    }


# ══════════════════════════════════════════════════════
# CALCUL DES MÉTRIQUES RAGAS — sans la librairie RAGAS
# ══════════════════════════════════════════════════════
# Note : on utilise notre propre LLM (Ollama) comme juge
# au lieu de l'API OpenAI que RAGAS utilise par défaut
# c'est plus simple et ça ne coûte rien

import httpx
from app.config import settings


async def compute_faithfulness(
    question: str,
    answer: str,
    contexts: list[str]
) -> float:
    """
    Mesure si la réponse est ancrée dans les documents.
    Score 0-1 : 1 = tout vient des docs, 0 = hallucination pure.

    On demande au LLM : "Est-ce que cette réponse est supportée
    par ces documents ?"

    question : la question posée
    answer   : la réponse générée par le système
    contexts : les chunks utilisés comme contexte
    -> float : score entre 0 et 1
    """

    # Construire le contexte combiné
    combined_context = "\n\n".join(contexts)

    # Prompt pour le LLM juge
    # On demande un score entre 0 et 1
    # 0 = réponse inventée, 1 = réponse entièrement ancrée dans les docs
    prompt = f"""You are an evaluator assessing if an answer is grounded in the provided context.

CONTEXT:
{combined_context}

QUESTION: {question}

ANSWER: {answer}

Rate the faithfulness of the answer on a scale from 0 to 1:
- 1.0 = every claim in the answer is directly supported by the context
- 0.5 = some claims are supported, some are not
- 0.0 = the answer contains claims not found in the context (hallucination)

Respond with ONLY a number between 0 and 1 (e.g., 0.8).
Score:"""

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"{settings.OLLAMA_BASE_URL}/api/generate",
            json={
                "model" : settings.LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    # temperature=0 → score déterministe
                    "temperature": 0.0,
                    "num_predict": 10
                }
            }
        )
        response.raise_for_status()

    # Extraire le score numérique de la réponse
    raw = response.json()["response"].strip()
    try:
        # Convertir en float — ex: "0.8" → 0.8
        score = float(raw.split()[0].rstrip("."))
        # S'assurer que le score est entre 0 et 1
        return max(0.0, min(1.0, score))
    except:
        # Si le LLM ne retourne pas un nombre → score neutre
        return 0.5


async def compute_answer_relevancy(
    question: str,
    answer: str
) -> float:
    """
    Mesure si la réponse répond vraiment à la question.
    Score 0-1 : 1 = réponse directe et complète, 0 = répond à côté.

    question : la question posée
    answer   : la réponse générée
    -> float : score entre 0 et 1
    """

    prompt = f"""You are an evaluator assessing if an answer is relevant to the question.

QUESTION: {question}

ANSWER: {answer}

Rate the relevancy of the answer on a scale from 0 to 1:
- 1.0 = the answer directly and completely addresses the question
- 0.5 = the answer partially addresses the question
- 0.0 = the answer does not address the question at all

Respond with ONLY a number between 0 and 1 (e.g., 0.8).
Score:"""

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"{settings.OLLAMA_BASE_URL}/api/generate",
            json={
                "model" : settings.LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 10}
            }
        )
        response.raise_for_status()

    raw = response.json()["response"].strip()
    try:
        score = float(raw.split()[0].rstrip("."))
        return max(0.0, min(1.0, score))
    except:
        return 0.5


def compute_context_precision_cosine(
    contexts: list[str],
    ground_truth: str
) -> float:
    """
    Mesure la pertinence des chunks par similarité cosine
    avec la ground truth — au lieu de demander au LLM.

    Avantages vs LLM-as-Judge :
    - Déterministe — toujours le même score
    - Pas de biais du petit LLM juge
    - Plus rapide — pas d'appel HTTP
    - Plus honnête — mesure réelle de similarité sémantique

    contexts     : les chunks récupérés par le retriever
    ground_truth : la vraie réponse attendue
    -> float     : proportion de chunks pertinents (score > seuil)
    """

    if not contexts:
        return 0.0

    # Importer le modèle d'embeddings déjà chargé en RAM
    # on réutilise nomic-embed-text-v1.5 — le même que le retriever
    # pas besoin de recharger un modèle — pattern Singleton
    from app.ingestion.embedder import model, DOCUMENT_PREFIX

    # Vectoriser la ground truth
    # on utilise le prefix "search_document:" — cohérent avec l'ingestion
    gt_vector = model.encode(
        [DOCUMENT_PREFIX + ground_truth],
        convert_to_numpy=True
    )[0]

    # Vectoriser chaque chunk
    chunk_vectors = model.encode(
        [DOCUMENT_PREFIX + ctx for ctx in contexts],
        convert_to_numpy=True
    )

    # Calculer la similarité cosine entre chaque chunk et la ground truth
    # similarité cosine = produit scalaire / (norme1 * norme2)
    # valeur entre -1 et 1 — plus proche de 1 = plus similaire
    useful_count = 0
    scores = []

    for chunk_vec in chunk_vectors:
        # Calcul cosine manuellement — pas besoin de sklearn
        dot_product = float(gt_vector @ chunk_vec)
        norm_gt     = float((gt_vector @ gt_vector) ** 0.5)
        norm_chunk  = float((chunk_vec @ chunk_vec) ** 0.5)

        if norm_gt > 0 and norm_chunk > 0:
            cosine_score = dot_product / (norm_gt * norm_chunk)
        else:
            cosine_score = 0.0

        scores.append(round(cosine_score, 3))

        # Seuil : chunk considéré pertinent si similarité > 0.5
        # 0.5 = seuil raisonnable pour nomic-embed-text-v1.5
        if cosine_score > 0.5:
            useful_count += 1

    # Afficher les scores pour debug
    print(f"    Scores cosine chunks : {scores}")

    # Proportion de chunks pertinents
    return useful_count / len(contexts)


# ══════════════════════════════════════════════════════
# PIPELINE D'ÉVALUATION COMPLET
# ══════════════════════════════════════════════════════

async def evaluate_strategy(
    strategy_name: str,
    strategy_func,
    dataset: list[dict]
) -> dict:
    """
    Évalue une stratégie RAG complète sur tout le dataset.
    Calcule les 3 métriques pour chaque question puis fait la moyenne.

    strategy_name : nom de la stratégie (ex: "dense", "hybrid")
    strategy_func : la fonction async à appeler (run_dense, run_hybrid...)
    dataset       : liste de {question, ground_truth}
    -> dict       : scores moyens pour cette stratégie
    """

    print(f"\n{'='*50}")
    print(f"Évaluation : {strategy_name}")
    print(f"{'='*50}")

    # Listes pour accumuler les scores de chaque question
    faithfulness_scores    = []
    relevancy_scores       = []
    precision_scores       = []

    # Évaluer chaque question du dataset
    for i, item in enumerate(dataset):
        question    = item["question"]
        ground_truth = item["ground_truth"]

        print(f"\nQuestion {i+1}/{len(dataset)} : {question[:60]}...")

        # ── Faire tourner la stratégie RAG ──
        # run_dense() ou run_hybrid() ou run_hybrid_reranker()
        result = await strategy_func(question)
        answer   = result["answer"]
        contexts = result["contexts"]

        print(f"  Réponse générée : {answer[:80]}...")

        # ── Calculer les 3 métriques ──
        faith = await compute_faithfulness(question, answer, contexts)
        relev = await compute_answer_relevancy(question, answer)
        prec = compute_context_precision_cosine(contexts, ground_truth)

        faithfulness_scores.append(faith)
        relevancy_scores.append(relev)
        precision_scores.append(prec)

        print(f"  Faithfulness    : {faith:.2f}")
        print(f"  Answer Relevancy: {relev:.2f}")
        print(f"  Context Precision: {prec:.2f}")

    # Calculer les moyennes sur toutes les questions
    avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)
    avg_relevancy    = sum(relevancy_scores)    / len(relevancy_scores)
    avg_precision    = sum(precision_scores)    / len(precision_scores)

    # Score global = moyenne des 3 métriques
    overall = (avg_faithfulness + avg_relevancy + avg_precision) / 3

    results = {
        "strategy"         : strategy_name,
        "faithfulness"     : round(avg_faithfulness, 3),
        "answer_relevancy" : round(avg_relevancy, 3),
        "context_precision": round(avg_precision, 3),
        "overall"          : round(overall, 3),

        # Garder les scores détaillés par question pour analyse
        "details": {
            "faithfulness_per_question"    : [round(s, 2) for s in faithfulness_scores],
            "relevancy_per_question"       : [round(s, 2) for s in relevancy_scores],
            "precision_per_question"       : [round(s, 2) for s in precision_scores],
        }
    }

    print(f"\n── Résultats moyens {strategy_name} ──")
    print(f"  Faithfulness     : {avg_faithfulness:.3f}")
    print(f"  Answer Relevancy : {avg_relevancy:.3f}")
    print(f"  Context Precision: {avg_precision:.3f}")
    print(f"  Overall          : {overall:.3f}")

    return results


async def run_all_evaluations():
    """
    Lance l'évaluation des 3 stratégies et sauvegarde les résultats.
    C'est la fonction principale — point d'entrée du script.
    """

    print("🚀 Démarrage de l'évaluation RAGAS")
    print(f"Dataset : {len(EVALUATION_DATASET)} questions")
    print("Stratégies : Dense | Hybride | Hybride+Reranker")

    # Évaluer les 3 stratégies une par une
    results = []

    # Stratégie 1 — Dense seul
    r1 = await evaluate_strategy(
        strategy_name="Dense",
        strategy_func=run_dense,
        dataset=EVALUATION_DATASET
    )
    results.append(r1)

    # Stratégie 2 — Hybride sans reranker
    r2 = await evaluate_strategy(
        strategy_name="Hybrid (BM25+RRF)",
        strategy_func=run_hybrid,
        dataset=EVALUATION_DATASET
    )
    results.append(r2)

    # Stratégie 3 — Hybride avec reranker
    r3 = await evaluate_strategy(
        strategy_name="Hybrid + Reranker",
        strategy_func=run_hybrid_reranker,
        dataset=EVALUATION_DATASET
    )
    results.append(r3)

    # Sauvegarder les résultats en JSON
    # utile pour les garder et les analyser plus tard
    output_path = "eval/results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Résultats sauvegardés dans {output_path}")

    return results


# Point d'entrée — exécuté quand on lance le script directement
if __name__ == "__main__":
    asyncio.run(run_all_evaluations())