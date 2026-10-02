# httpx pour appeler Ollama
import httpx

# settings pour l'URL d'Ollama
from app.config import settings


async def classify_intent(question: str) -> str:
    """
    Classifie la question en 3 catégories :
    - "rag_search"    : question sur les docs techniques
    - "direct_answer" : question générale sur le Data Engineering
    - "chitchat"      : conversation simple, salutation

    On utilise le LLM pour classifier — plus robuste qu'un système de règles.
    Le LLM comprend le contexte et les nuances de la question.
    """

    # ── Prompt de classification ──
    # On demande au LLM de retourner UN SEUL MOT
    # temperature=0 → toujours la même classification pour la même question
    # num_predict=5 → on veut juste un mot, pas une phrase
    classification_prompt = f"""You are a question classifier for a Data Engineering assistant.
Classify the following question into exactly ONE category.

Categories:
- rag_search: Technical questions about specific pipelines, errors, configurations,
  incidents, or anything that requires searching internal documentation
  Examples: "Why does Spark fail?", "Airflow DAG errors", "pipeline configuration"

- direct_answer: General knowledge questions about Data Engineering concepts
  that don't require internal documentation
  Examples: "What is Spark?", "Explain MapReduce", "What is a DAG?"

- chitchat: Greetings, small talk, or non-technical questions
  Examples: "Hello!", "Thank you", "How are you?", "What time is it?"

Question: "{question}"

Respond with ONLY one word: rag_search, direct_answer, or chitchat"""

    # Appel à Ollama pour la classification
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{settings.OLLAMA_BASE_URL}/api/generate",
            json={
                "model" : settings.LLM_MODEL,
                "prompt": classification_prompt,
                "stream": False,
                "options": {
                    # temperature=0 → classification déterministe
                    # toujours le même résultat pour la même question
                    "temperature": 0.0,

                    # num_predict=10 → on attend juste un mot
                    # évite que le LLM explique son choix
                    "num_predict": 10
                }
            }
        )
        response.raise_for_status()

    # Extraire la classification
    raw = response.json()["response"].strip().lower()

    # Normaliser la réponse — le LLM peut répondre avec des variations
    # ex: "rag_search." ou "RAG_SEARCH" ou "rag search"
    if "rag" in raw:
        return "rag_search"
    elif "direct" in raw:
        return "direct_answer"
    elif "chit" in raw or "chat" in raw:
        return "chitchat"
    else:
        # Par défaut → rag_search est le comportement le plus utile
        # si le LLM retourne quelque chose d'inattendu
        return "rag_search"


def route_after_intent(state: dict) -> str:
    """
    Edge conditionnel — décide quel nœud exécuter
    après analyze_intent selon l'intent classifié.

    LangGraph appelle cette fonction après chaque exécution
    de analyze_intent pour décider où aller.

    state : l'état actuel du graphe
    -> str : le nom du prochain nœud à exécuter
    """

    intent = state["intent"]

    # Routing selon l'intent
    if intent == "rag_search":
        # Question technique → cherche dans les docs
        return "rag_search"
    elif intent == "direct_answer":
        # Question générale → LLM seul sans RAG
        return "direct_answer"
    else:
        # Chitchat ou autre → réponse courte directe
        return "chitchat"