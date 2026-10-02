# On importe les outils des sprints précédents
# L'agent réutilise tout ce qu'on a construit — rien n'est perdu
from app.rag.hybrid_retriever import hybrid_retrieve
from app.rag.reranker import rerank
from app.rag.generator import generate, build_prompt
from app.llm.client import ollama_client
from app.agent.router import classify_intent
import httpx
from app.config import settings


async def analyze_intent_node(state: dict) -> dict:
    """
    Nœud 1 — Analyse l'intent de la question.
    Classifie en : rag_search / direct_answer / chitchat.

    Reçoit : {question, steps, ...}
    Retourne : mise à jour de l'état avec intent
    """

    question = state["question"]

    # Classifier la question avec le LLM
    intent = await classify_intent(question)

    # On retourne UNIQUEMENT les champs qu'on met à jour
    # LangGraph fusionne ce dict avec l'état existant
    # les autres champs (question, sources...) restent inchangés
    return {
        # L'intent classifié — guidera le routing
        "intent": intent,

        # Ajouter cette étape à la liste des étapes
        # state["steps"] + [...] crée une nouvelle liste
        # on ne modifie pas l'état directement — LangGraph gère ça
        "steps": state["steps"] + ["analyze_intent"]
    }


async def rag_search_node(state: dict) -> dict:
    """
    Nœud 2a — Recherche RAG hybride + génération.
    Utilisé quand intent == "rag_search".
    Réutilise le pipeline complet du Sprint 4.

    Reçoit : {question, intent, steps, ...}
    Retourne : mise à jour avec chunks, answer, sources
    """

    question = state["question"]

    # ── Retrieval hybride (Sprint 4) ──
    # Dense + BM25 + RRF → top-6 chunks
    chunks = hybrid_retrieve(question, k=6)

    # CrossEncoder reranking → top-3 chunks les plus pertinents
    reranked_chunks = await rerank(question, chunks, top_k=3)

    # ── Génération ──
    # Construire le prompt RAG et appeler le LLM
    answer = await generate(question, reranked_chunks)

    # ── Construire les sources ──
    sources = []
    for chunk in reranked_chunks:
        sources.append({
            "source": chunk["payload"]["source"],
            "page"  : chunk["payload"].get("page", 1),
            "score" : round(chunk["score"], 2),
            "chunk" : chunk["payload"]["text"][:150] + "..."
        })

    return {
        "chunks" : reranked_chunks,
        "answer" : answer,
        "sources": sources,
        "steps"  : state["steps"] + ["rag_search"]
    }


async def direct_answer_node(state: dict) -> dict:
    """
    Nœud 2b — Réponse directe sans RAG.
    Utilisé pour les questions générales sur le Data Engineering.
    Appelle directement le LLM sans chercher dans Qdrant.

    Reçoit : {question, intent, steps, ...}
    Retourne : mise à jour avec answer
    """

    question = state["question"]

    # Prompt adapté pour les questions générales
    # pas de contexte RAG — le LLM utilise sa connaissance générale
    prompt = f"""You are an expert Data Engineering assistant.
Answer the following general Data Engineering question clearly and concisely.
Use your knowledge to provide a helpful, accurate answer.

Question: {question}

Answer:"""

    # Appel direct au LLM via Ollama
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"{settings.OLLAMA_BASE_URL}/api/generate",
            json={
                "model"  : settings.LLM_MODEL,
                "prompt" : prompt,
                "stream" : False,
                "options": {
                    # temperature légèrement plus haute qu'en RAG
                    # on veut une réponse plus naturelle et explicative
                    "temperature": 0.3,
                    "num_predict": 512
                }
            }
        )
        response.raise_for_status()
        answer = response.json()["response"]

    return {
        "answer" : answer,
        # Pas de sources — réponse directe sans documents
        "sources": [],
        "steps"  : state["steps"] + ["direct_answer"]
    }


async def chitchat_node(state: dict) -> dict:
    """
    Nœud 2c — Réponse conversationnelle courte.
    Utilisé pour les salutations et questions hors sujet.

    Reçoit : {question, intent, steps, ...}
    Retourne : mise à jour avec answer
    """

    question = state["question"]

    # Prompt conversationnel — court et amical
    prompt = f"""You are a friendly Data Engineering assistant.
Respond briefly and naturally to this message.
If it's a greeting, greet back. If it's off-topic, politely redirect
to Data Engineering topics.

Message: {question}

Response:"""

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{settings.OLLAMA_BASE_URL}/api/generate",
            json={
                "model"  : settings.LLM_MODEL,
                "prompt" : prompt,
                "stream" : False,
                "options": {
                    # temperature plus haute pour des réponses naturelles
                    "temperature": 0.7,

                    # réponse courte — c'est du chitchat
                    "num_predict": 150
                }
            }
        )
        response.raise_for_status()
        answer = response.json()["response"]

    return {
        "answer" : answer,
        "sources": [],
        "steps"  : state["steps"] + ["chitchat"]
    }


async def format_response_node(state: dict) -> dict:
    """
    Nœud 3 — Formate la réponse finale.
    Dernier nœud avant END — tous les chemins y arrivent.
    Peut enrichir ou nettoyer la réponse si nécessaire.

    Reçoit : l'état complet avec answer, sources, intent, steps
    Retourne : mise à jour finale de l'état
    """

    # Pour l'instant on formate simplement
    # Dans les sprints suivants on pourrait :
    # - ajouter un résumé des sources
    # - formater en markdown
    # - détecter et corriger les hallucinations

    return {
        # Marquer l'étape finale
        "steps": state["steps"] + ["format_response"]
    }