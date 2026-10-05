# httpx pour appeler Ollama via HTTP
import httpx

# settings pour récupérer OLLAMA_BASE_URL et LLM_MODEL depuis le .env
from app.config import settings


# ══════════════════════════════════════════════════════
# COUCHE 1 — Règles heuristiques
# ══════════════════════════════════════════════════════

# Dictionnaire des mots-clés par intent
# Chaque liste contient des signaux FORTS et CERTAINS
# Si un de ces mots apparaît → on est sûr à 95%+ de l'intent
# On n'appelle PAS le LLM dans ce cas → plus rapide + déterministe

INTENT_KEYWORDS = {

    # sql_query — signaux forts : l'utilisateur veut des DONNÉES d'une table
    # "show me", "list", "get", "count" → clairement une requête de données
    # Note : "failed pipelines" sans "why" → données, pas logs
    "sql_query": [
        "show me",       # "show me all failed pipelines"
        "list all",      # "list all customers"
        "list the",      # "list the failed runs"
        "get all",       # "get all records"
        "get me",        # "get me the errors"
        "count",         # "count the failed tasks"
        "how many",      # "how many pipelines failed"
        "fetch",         # "fetch all records"
        "select",        # "select * from..."
        "retrieve",      # "retrieve all data"
        "give me all",   # "give me all the pipelines"
        "find all",      # "find all failed jobs"
        "return all",    # "return all records"
    ],

    # db_inspect — signaux forts : l'utilisateur veut voir la STRUCTURE de la DB
    # pas les données — juste les tables, colonnes, types
    "db_inspect": [
        "what tables",   # "what tables are in the database"
        "list tables",   # "list tables in the db"
        "show tables",   # "show all tables"
        "schema",        # "show me the schema"
        "what columns",  # "what columns does the table have"
        "describe",      # "describe the customers table"
        "structure of",  # "structure of the pipeline table"
        "database structure",
        "table structure",
    ],

    # log_analysis — signaux forts : l'utilisateur veut COMPRENDRE une erreur
    # "why", "what caused", "analyze" → clairement une analyse de logs
    "log_analysis": [
        "why did",       # "why did the pipeline fail"
        "why does",      # "why does it crash"
        "what caused",   # "what caused the error"
        "analyze log",   # "analyze the logs"
        "analyse log",   # variante française
        "root cause",    # "root cause of the failure"
        "what went wrong",
        "error in log",  # "errors in the logs"
        "errors in log",
    ],

    # chitchat — signaux forts : conversation simple, jamais ambigu
    "chitchat": [
        "hello",         # "hello !"
        "hi ",           # "hi there" (espace pour éviter "high")
        "hi!",
        "thank you",     # "thank you"
        "thanks",        # "thanks!"
        "good morning",
        "good afternoon",
        "how are you",   # "how are you ?"
        "what's up",
        "bye",
        "goodbye",
    ],
}


def pre_classify(question: str) -> str | None:
    """
    COUCHE 1 — Classification par règles heuristiques.
    Rapide, déterministe, sans appel LLM.

    Cherche des mots-clés évidents dans la question.
    Si trouvé → retourne l'intent directement.
    Si rien trouvé → retourne None → la COUCHE 2 (LLM) prend le relais.

    question : la question de l'utilisateur
    -> str | None : l'intent si certain, None si incertain
    """

    # Normaliser la question — tout en minuscules + strip
    # "Show Me ALL" → "show me all"
    # important pour que les comparaisons fonctionnent
    q = question.lower().strip()

    # Parcourir chaque intent et ses mots-clés
    for intent, keywords in INTENT_KEYWORDS.items():
        for keyword in keywords:
            # Vérifier si le mot-clé est présent dans la question
            # "in" cherche n'importe où dans le texte
            # ex: "show me all failed" contient "show me" → sql_query
            if keyword in q:
                # Mot-clé trouvé → on est certain de l'intent
                # Retourner immédiatement sans appeler le LLM
                return intent

    # Aucun mot-clé trouvé → cas incertain
    # La couche 2 (LLM) va prendre le relais
    return None


# ══════════════════════════════════════════════════════
# COUCHE 2 — LLM avec prompt amélioré
# ══════════════════════════════════════════════════════

async def classify_with_llm(question: str) -> str:
    """
    COUCHE 2 — Classification par LLM avec prompt amélioré.
    Appelée uniquement quand la couche 1 n'est pas certaine.

    Améliorations par rapport à l'ancien prompt :
    1. Contre-exemples explicites (NOT this)
    2. Signaux clés (KEY signal) pour guider le LLM
    3. Ordre des catégories du plus spécifique au plus général
    4. Exemples plus variés et plus proches des vrais cas

    question : la question passée par la couche 1
    -> str   : l'intent classifié par le LLM
    """

    classification_prompt = f"""You are a question classifier for a Data Engineering assistant.
Classify the following question into exactly ONE category.

Categories:

- sql_query: The user wants to RETRIEVE ACTUAL DATA from a database table using SQL.
  KEY signals: "show me", "list", "get", "count", "how many", "fetch"
  Examples: "Show me the last 5 failed runs",
            "How many records in customers?",
            "Get all pipelines where status is failed",
            "Count the errors in the logs table"
  NOT this: questions asking WHY something failed (that is log_analysis)
  NOT this: questions about table structure or columns (that is db_inspect)

- db_inspect: The user wants to see the DATABASE STRUCTURE (tables, columns, types).
  KEY signals: "what tables", "schema", "columns", "describe", "structure"
  Examples: "What tables are in the database?",
            "What columns does the customers table have?",
            "Describe the pipeline table"
  NOT this: questions asking to retrieve actual data rows (that is sql_query)

- log_analysis: The user wants to ANALYZE LOGS to understand errors or failures.
  KEY signals: "why", "what caused", "analyze", "root cause", "what went wrong"
  Examples: "Why did the pipeline fail last night?",
            "What errors appeared in the logs?",
            "What caused the ETL crash?"
  NOT this: questions asking to list or count records from a table (that is sql_query)

- rag_search: The user wants to search INTERNAL DOCUMENTATION for technical answers.
  KEY signals: questions about configuration, best practices, how-to
  Examples: "Why does Spark fail with OutOfMemoryError?",
            "How to configure Airflow DAGs?",
            "What is the recommended Spark memory setting?"
  NOT this: questions about actual data in the database (that is sql_query)

- direct_answer: General Data Engineering KNOWLEDGE questions (no docs needed).
  KEY signals: "what is", "explain", "define", "how does X work"
  Examples: "What is Apache Spark?",
            "Explain MapReduce",
            "What is a DAG?"

- chitchat: Greetings or non-technical conversation.
  Examples: "Hello!", "Thank you", "How are you?"

Question: "{question}"

Respond with ONLY one word: sql_query, db_inspect, log_analysis, rag_search, direct_answer, or chitchat"""

    # Appel au LLM — uniquement pour les cas incertains
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{settings.OLLAMA_BASE_URL}/api/generate",
            json={
                "model" : settings.LLM_MODEL,
                "prompt": classification_prompt,
                "stream": False,
                "options": {
                    # temperature=0.0 → classification déterministe
                    "temperature": 0.0,
                    # num_predict=10 → un seul mot attendu
                    "num_predict": 10
                }
            }
        )
        response.raise_for_status()

    # Extraire et normaliser la réponse du LLM
    raw = response.json()["response"].strip().lower()

    # Normalisation — gère les variations de réponse du LLM
    # ordre important : du plus spécifique au plus général
    if "sql" in raw or "query" in raw:
        return "sql_query"
    elif "log" in raw:
        return "log_analysis"
    elif "db" in raw or "inspect" in raw or "schema" in raw:
        return "db_inspect"
    elif "rag" in raw or "search" in raw:
        return "rag_search"
    elif "direct" in raw or "answer" in raw:
        return "direct_answer"
    elif "chit" in raw or "chat" in raw:
        return "chitchat"
    else:
        # Défaut → rag_search
        return "rag_search"


# ══════════════════════════════════════════════════════
# FONCTION PRINCIPALE — combine les deux couches
# ══════════════════════════════════════════════════════

async def classify_intent(question: str) -> str:
    """
    Classification en deux couches :

    COUCHE 1 — Règles heuristiques (rapide, déterministe)
        Si mots-clés évidents trouvés → retourne l'intent directement
        Avantage : zéro appel LLM, toujours correct pour les cas évidents

    COUCHE 2 — LLM avec prompt amélioré (pour les cas ambigus)
        Si la couche 1 n'est pas certaine → appelle le LLM
        Prompt amélioré avec contre-exemples pour les cas ambigus

    question : la question de l'utilisateur
    -> str   : l'intent final classifié
    """

    # ── Couche 1 : essayer les règles d'abord ──
    # pre_classify() cherche des mots-clés évidents
    # si trouvé → retourne l'intent sans appeler le LLM
    intent = pre_classify(question)

    if intent is not None:
        # Mot-clé certain trouvé → retourner directement
        # "show me all failed pipelines" → sql_query ⚡ (pas de LLM)
        return intent

    # ── Couche 2 : LLM pour les cas incertains ──
    # pre_classify() a retourné None → cas ambigu
    # on appelle le LLM avec le prompt amélioré
    # "Why does Spark fail?" → pas de mot-clé évident → LLM → rag_search
    return await classify_with_llm(question)


# ══════════════════════════════════════════════════════
# FONCTIONS DE ROUTING — inchangées
# ══════════════════════════════════════════════════════

def route_after_intent(state: dict) -> str:
    """
    Edge conditionnel principal — appelé par LangGraph après analyze_intent.
    Retourne le nom du nœud à exécuter selon l'intent classifié.
    """
    intent = state["intent"]

    if intent == "log_analysis":
        return "log_analyzer"
    elif intent == "db_inspect":
        return "db_inspect"
    elif intent == "sql_query":
        return "sql_query"
    elif intent == "rag_search":
        return "rag_search"
    elif intent == "direct_answer":
        return "direct_answer"
    else:
        return "chitchat"


def route_after_sql(state: dict) -> str:
    """
    Edge conditionnel SQL — après sql_query_node.
    Gère le Human in the Loop.
    """
    sql_approved = state.get("sql_approved")

    if sql_approved is True:
        return "sql_execute_node"
    elif sql_approved is False:
        return "format_response"
    else:
        # None → graphe s'interrompt grâce à interrupt_before
        return "sql_execute_node"