# httpx pour appeler Ollama via HTTP
# même pattern que dans tous les autres fichiers du projet
import httpx

# settings pour récupérer OLLAMA_BASE_URL et LLM_MODEL depuis le .env
from app.config import settings


async def classify_intent(question: str) -> str:
    """
    Classifie la question en 6 catégories d'intent.
    Utilise le LLM avec temperature=0.0 pour une classification déterministe.
    C'est du zero-shot classification — pas d'exemples d'entraînement,
    le LLM comprend directement les catégories décrites en langage naturel.

    question : la question de l'utilisateur
    -> str   : l'intent classifié parmi les 6 catégories
    """

    # ── Prompt de classification zero-shot ──
    # On décrit chaque catégorie avec des exemples concrets
    # Le LLM doit retourner UN SEUL MOT parmi les 6 catégories
    # num_predict=10 → on attend juste un mot, pas une phrase
    # temperature=0.0 → toujours la même classification pour la même question
    classification_prompt = f"""You are a question classifier for a Data Engineering assistant.
Classify the following question into exactly ONE category.

Categories:
- rag_search: Technical questions requiring internal documentation search
  Examples: "Why does Spark fail?", "Airflow DAG errors", "pipeline configuration",
            "How to fix OutOfMemoryError?", "What is the recommended Spark config?"

- db_inspect: Questions about database structure, tables, schemas, columns
  Examples: "What tables are in the database?", "Show me the schema",
            "What columns does the customers table have?",
            "Describe the pipeline table", "List all tables"

- sql_query: Questions requiring SQL execution to retrieve actual data
  Examples: "Show me the last 5 failed runs", "How many records in customers?",
            "What pipelines ran today?", "Count the errors in the logs table",
            "Show me all failed tasks"

- log_analysis: Questions about pipeline logs, errors, failures, incidents
  Examples: "Why did the pipeline fail last night?", "What errors appeared in logs?",
            "Show me recent pipeline failures", "Analyze the ETL errors",
            "What caused the pipeline crash?"

- direct_answer: General Data Engineering knowledge questions
  Examples: "What is Spark?", "Explain MapReduce", "What is a DAG?",
            "What is ETL?", "Explain partitioning"

- chitchat: Greetings, small talk, or non-technical conversation
  Examples: "Hello!", "Thank you", "How are you?", "What time is it?"

Question: "{question}"

Respond with ONLY one word: rag_search, db_inspect, sql_query, log_analysis, direct_answer, or chitchat"""

    # ── Appel à Ollama pour la classification ──
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{settings.OLLAMA_BASE_URL}/api/generate",
            json={
                "model" : settings.LLM_MODEL,
                "prompt": classification_prompt,
                "stream": False,
                "options": {
                    # temperature=0.0 → classification 100% déterministe
                    # toujours le même résultat pour la même question
                    # crucial pour la fiabilité du routing
                    "temperature": 0.0,

                    # num_predict=10 → on attend juste un mot
                    # évite que le LLM explique son choix
                    "num_predict": 10
                }
            }
        )
        response.raise_for_status()

    # Extraire et normaliser la réponse du LLM
    # strip() → enlever les espaces et sauts de ligne
    # lower() → tout en minuscules pour la comparaison
    raw = response.json()["response"].strip().lower()

    # ── Normalisation robuste ──
    # Le LLM peut répondre avec des variations :
    # "LOG_ANALYSIS." ou "log analysis" ou "log_analysis"
    # On utilise "in" pour capturer toutes les variations
    # L'ORDRE DES CONDITIONS EST IMPORTANT — du plus spécifique au plus général

    # log_analysis — vérifié EN PREMIER car "log" peut apparaître dans d'autres contextes
    if "log" in raw or "log_analysis" in raw:
        return "log_analysis"

    # db_inspect — mots clés liés à la structure de la DB
    elif "db" in raw or "inspect" in raw or "schema" in raw or "database" in raw:
        return "db_inspect"

    # sql_query — mots clés liés à l'exécution SQL
    elif "sql" in raw or "query" in raw:
        return "sql_query"

    # rag_search — mots clés liés à la recherche documentaire
    elif "rag" in raw or "search" in raw:
        return "rag_search"

    # direct_answer — mots clés liés aux réponses générales
    elif "direct" in raw or "answer" in raw:
        return "direct_answer"

    # chitchat — mots clés liés à la conversation
    elif "chit" in raw or "chat" in raw:
        return "chitchat"

    else:
        # Défaut → rag_search si le LLM retourne quelque chose d'inattendu
        # rag_search est le comportement le plus utile par défaut
        # plutôt que de laisser l'agent sans réponse
        return "rag_search"


def route_after_intent(state: dict) -> str:
    """
    Edge conditionnel principal — appelé par LangGraph après analyze_intent.
    Décide vers quel nœud aller selon l'intent classifié dans l'état.

    Cette fonction n'est PAS un nœud — LangGraph l'appelle automatiquement
    après analyze_intent_node pour décider la prochaine étape.

    state : l'état actuel du graphe (contient intent classifié)
    -> str : le nom du prochain nœud à exécuter
    """

    intent = state["intent"]

    # ── Routing vers les 6 nœuds possibles ──
    if intent == "log_analysis":
        # Question sur les logs → Log Analyzer
        # parse les logs, extrait les erreurs, synthèse LLM
        return "log_analyzer"

    elif intent == "db_inspect":
        # Question sur la structure DB → DB Inspector
        # inspecte PostgreSQL via information_schema
        return "db_inspect"

    elif intent == "sql_query":
        # Question nécessitant des données → SQL Query Node
        # génère la requête SQL puis attend confirmation humaine
        return "sql_query"

    elif intent == "rag_search":
        # Question technique → RAG hybride Sprint 4
        # Dense + BM25 + RRF + CrossEncoder
        return "rag_search"

    elif intent == "direct_answer":
        # Question générale → LLM seul sans RAG
        # utilise la connaissance générale du modèle
        return "direct_answer"

    else:
        # Chitchat ou intent inconnu → réponse conversationnelle
        # temperature=0.7, réponse courte et naturelle
        return "chitchat"


def route_after_sql(state: dict) -> str:
    """
    Edge conditionnel SQL — appelé par LangGraph après sql_query_node.
    Gère le flux du Human in the Loop pour l'exécution SQL.

    3 cas possibles selon sql_approved dans l'état :

    sql_approved = True  → l'humain a approuvé → exécuter la requête
    sql_approved = False → l'humain a refusé   → aller à format_response
    sql_approved = None  → pas encore décidé   → aller à sql_execute
                           (LangGraph va s'interrompre grâce à interrupt_before)

    state : l'état actuel (contient sql_approved)
    -> str : "sql_execute" ou "format_response"
    """

    sql_approved = state.get("sql_approved")

    if sql_approved is True:
        # ── Humain a approuvé ──
        # Deux cas possibles :
        # 1. Première exécution avec approbation immédiate (rare)
        # 2. Reprise après interruption avec sql_approved=True
        # Dans les deux cas → exécuter la requête SQL
        return "sql_execute_node"

    elif sql_approved is False:
        # ── Humain a refusé ──
        # sql_query_node a déjà mis dans l'état :
        # answer = "SQL execution was not approved"
        # On saute sql_execute et on va directement formatter la réponse
        return "format_response"

    else:
        # ── sql_approved est None — pas encore décidé ──
        # On retourne "sql_execute" mais LangGraph va s'interrompre
        # AVANT de l'exécuter grâce à interrupt_before=["sql_execute"]
        # dans la compilation du graphe (graph.py)
        #
        # L'état est sauvegardé dans MemorySaver avec le thread_id
        # L'humain voit la requête SQL dans la réponse de l'API
        # Il renvoie une requête avec sql_approved=True ou False
        # Le graphe reprend depuis sql_execute
        return "sql_execute_node"