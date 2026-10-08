# ── Imports Sprint 4 — RAG pipeline ──
from app.rag.hybrid_retriever import hybrid_retrieve
from app.rag.reranker import rerank
from app.rag.generator import generate   # build_prompt non utilisé ici

# ── Imports Sprint 5 — Router ──
from app.agent.router import classify_intent

# ── Imports Sprint 6 — Outils spécialisés ──
from app.agent.tools.db_inspector import inspect_database
from app.agent.tools.sql_executor import validate_query, execute_query
from app.agent.tools.log_analyzer import analyze_log_file, analyze_log_string

# ── Imports Python standard ──
from pathlib import Path
import httpx
from app.config import settings

from app.agent.tools.sql_utils import extract_sql


async def analyze_intent_node(state: dict) -> dict:
    """
    Nœud 1 — Analyse l'intent de la question.
    Classifie en 6 catégories : rag_search / direct_answer / chitchat
                                / db_inspect / sql_query / log_analysis
    """
    question = state["question"]

    # Classifier la question avec le LLM (temperature=0.0 → déterministe)
    intent = await classify_intent(question)

    # On retourne UNIQUEMENT les champs mis à jour
    # LangGraph fusionne avec l'état existant automatiquement
    return {
        "intent": intent,
        "steps" : state["steps"] + ["analyze_intent"]
    }


async def rag_search_node(state: dict) -> dict:
    """
    Nœud 2a — Recherche RAG hybride Sprint 4.
    Utilisé quand intent == "rag_search".
    Dense + BM25 + RRF + CrossEncoder + génération.
    """
    question = state["question"]

    # Retrieval hybride → top-6 chunks fusionnés
    chunks = hybrid_retrieve(question, k=6)

    # CrossEncoder reranking → top-3 chunks les plus pertinents
    reranked_chunks = await rerank(question, chunks, top_k=3)

    # Génération avec le prompt RAG structuré
    answer = await generate(question, reranked_chunks)

    # Construire les sources à retourner à l'utilisateur
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
    Le LLM répond avec sa connaissance générale — pas de Qdrant.
    temperature=0.3 → réponse naturelle et explicative.
    """
    question = state["question"]

    prompt = f"""You are an expert Data Engineering assistant.
Answer the following general Data Engineering question clearly and concisely.
Use your knowledge to provide a helpful, accurate answer.

Question: {question}

Answer:"""

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"{settings.OLLAMA_BASE_URL}/api/generate",
            json={
                "model"  : settings.LLM_MODEL,
                "prompt" : prompt,
                "stream" : False,
                "options": {"temperature": 0.3, "num_predict": 512}
            }
        )
        response.raise_for_status()
        answer = response.json()["response"]

    return {
        "answer" : answer,
        "sources": [],   # pas de sources — réponse depuis la connaissance générale
        "steps"  : state["steps"] + ["direct_answer"]
    }


async def chitchat_node(state: dict) -> dict:
    """
    Nœud 2c — Réponse conversationnelle courte.
    Utilisé pour les salutations et questions hors sujet.
    temperature=0.7 → réponse créative et naturelle.
    num_predict=150 → réponse courte — c'est du chitchat.
    """
    question = state["question"]

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
                "options": {"temperature": 0.7, "num_predict": 150}
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
    Nœud final — tous les chemins y convergent avant END.
    Marque l'étape finale dans steps[].
    Point d'extension pour le formatage futur (markdown, résumé sources...).
    """
    return {
        "steps": state["steps"] + ["format_response"]
    }


async def db_inspect_node(state: dict) -> dict:
    """
    Nœud DB Inspector — inspecte la structure de PostgreSQL.
    Utilisé quand intent == "db_inspect".
    Appelle inspect_database() → récupère tables + colonnes via information_schema.
    Passe les infos au LLM pour générer une réponse claire.
    temperature=0.1 → réponse factuelle ancrée dans la structure réelle de la DB.
    """
    question = state["question"]

    # Inspecter toutes les tables et leurs colonnes via information_schema
    db_info = inspect_database()

    # Formater les infos DB de façon lisible pour le LLM
    # "table_name (col1 type, col2 type, ...)"
    tables_summary = ""
    for table in db_info["tables"]:
        cols = ", ".join([
            f"{c['column_name']} ({c['data_type']})"
            for c in table["columns"]
        ])
        tables_summary += f"\nTable: {table['table_name']}\nColumns: {cols}\n"

    prompt = f"""You are a Data Engineering assistant with access to a PostgreSQL database.
Based on the database structure below, answer the user's question clearly.

DATABASE: {db_info['database']}
TABLES FOUND: {db_info['tables_count']}
{tables_summary}

QUESTION: {question}

ANSWER:"""

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"{settings.OLLAMA_BASE_URL}/api/generate",
            json={
                "model"  : settings.LLM_MODEL,
                "prompt" : prompt,
                "stream" : False,
                "options": {"temperature": 0.1, "num_predict": 512}
            }
        )
        response.raise_for_status()
        answer = response.json()["response"]

    return {
        "db_info": db_info,
        "answer" : answer,
        "sources": [],
        "steps"  : state["steps"] + ["db_inspect"]
    }


async def sql_query_node(state: dict) -> dict:
    """
    Nœud SQL Query — génère la requête SQL depuis la question.
    Améliorations Phase 7 :
    - extract_sql() robuste pour extraire la requête du texte LLM
    - Historique conversationnel pour gérer "fix the query"
    - Prompt amélioré sans "SQL Query (SELECT only):" qui était recopié
    """
    question = state["question"]

    # Récupérer la structure DB pour que le LLM génère du SQL correct
    db_info = inspect_database()
    tables_summary = "\n".join([
        f"- {t['table_name']}: " +
        ", ".join([c['column_name'] for c in t['columns']])
        for t in db_info["tables"]
    ])

    # ── Récupérer l'historique conversationnel ──
    # Permet de gérer "fix the query" ou "it's a select query"
    # Sans historique → l'agent invente une nouvelle requête sans rapport
    history = state.get("history", [])
    history_text = "\n".join(
        f"{m['role']}: {m['content'][:500]}"
        for m in history[-6:]   # 6 derniers messages max
    )

    # ── Prompt amélioré ──
    # IMPORTANT : ne pas terminer par "SQL Query (SELECT only):"
    # Le LLM recopiait cette phrase dans sa réponse → extract_sql la gérait mais
    # c'était une source d'erreur. On termine par "SQL:" pour guider sans polluer.
    sql_prompt = f"""You are a PostgreSQL expert. Write ONE read-only SQL query.
Return ONLY the SQL query. No explanation, no markdown, no labels.

Available tables and columns:
{tables_summary}

Conversation so far:
{history_text or '(none)'}

Question: {question}

Rules:
- Only SELECT queries are allowed.
- If the user asks to fix or correct the query, correct the last SQL query from the conversation.
- To count tables use: SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'

SQL:"""

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"{settings.OLLAMA_BASE_URL}/api/generate",
            json={
                "model"  : settings.LLM_MODEL,
                "prompt" : sql_prompt,
                "stream" : False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 300   # ← augmenté pour les requêtes complexes
                }
            }
        )
        response.raise_for_status()

    # ── B2 : Extraction robuste via extract_sql ──
    # Gère : texte avant, backticks, texte après, double instruction
    sql_query = extract_sql(response.json()["response"])

    # Si extract_sql retourne vide → le LLM n'a pas généré de SELECT
    if not sql_query:
        return {
            "sql_query"   : "",
            "sql_approved": False,
            "answer"      : "I could not generate a valid SQL query. Please rephrase your question.",
            "sources"     : [],
            "steps"       : state["steps"] + ["sql_query", "sql_rejected"]
        }

    # Valider la sécurité AVANT de demander confirmation
    is_safe, error_msg = validate_query(sql_query)

    if not is_safe:
        return {
            "sql_query"   : sql_query,
            "sql_approved": False,
            "answer"      : f"I cannot execute this query for security reasons: {error_msg}",
            "sources"     : [],
            "steps"       : state["steps"] + ["sql_query", "sql_rejected"]
        }

    return {
        "sql_query"   : sql_query,
        "sql_approved": None,
        "answer"      : f"I need your approval to execute this SQL query:\n\n{sql_query}",
        "sources"     : [],
        "steps"       : state["steps"] + ["sql_query"]
    }


async def sql_execute_node(state: dict) -> dict:
    """
    Nœud SQL Execute — exécute la requête APRÈS confirmation humaine.
    Ce nœud est exécuté lors du 2ème appel à ainvoke(None, config).

    Reçoit : état avec sql_approved=True et sql_query validée
    Étapes :
    1. Vérifie que sql_approved est bien True
    2. Exécute la requête via execute_query()
    3. Demande au LLM de formuler la réponse depuis les résultats
    """
    sql_query    = state["sql_query"]
    sql_approved = state.get("sql_approved")

    # Sécurité — ne s'exécute que si explicitement approuvé
    if not sql_approved:
        return {
            "answer": "SQL execution was not approved.",
            "steps" : state["steps"] + ["sql_execute_denied"]
        }

    # Exécuter la requête validée et approuvée
    result = execute_query(sql_query)

    if not result["success"]:
        # Erreur PostgreSQL — syntaxe incorrecte, table inexistante...
        return {
            "sql_result": result,
            "answer"    : f"SQL execution failed: {result['error']}",
            "sources"   : [],
            "steps"     : state["steps"] + ["sql_execute_error"]
        }

    # Formater les résultats pour le LLM
    # on prend les 10 premières lignes pour ne pas saturer le contexte
    rows_text = "\n".join([str(row) for row in result["rows"][:10]])

    answer_prompt = f"""Based on this SQL query result, answer the user's question.

Query executed: {sql_query}
Results ({result['row_count']} rows):
{rows_text}

Original question: {state['question']}

Answer:"""

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"{settings.OLLAMA_BASE_URL}/api/generate",
            json={
                "model"  : settings.LLM_MODEL,
                "prompt" : answer_prompt,
                "stream" : False,
                "options": {"temperature": 0.1, "num_predict": 512}
            }
        )
        response.raise_for_status()
        answer = response.json()["response"]

    return {
        "sql_result": result,
        "answer"    : answer,
        "sources"   : [],
        "steps"     : state["steps"] + ["sql_execute"]
    }


async def log_analyzer_node(state: dict) -> dict:
    """
    Nœud Log Analyzer — analyse les logs de pipeline.
    Utilisé quand intent == "log_analysis".

    Étapes :
    1. Cherche les fichiers .log dans ./logs/
    2. Si trouvés → analyse le plus récent
    3. Si absents → utilise un exemple de démonstration
    4. Construit un résumé structuré (erreurs, patterns, critiques)
    5. LLM synthétise les causes et solutions
    """
    question = state["question"]

    # Chercher les fichiers de logs dans ./logs/
    logs_dir = Path("./logs")

    if logs_dir.exists():
        log_files = list(logs_dir.glob("*.log"))

        if log_files:
            # Analyser le fichier le plus récent
            # st_mtime = timestamp de dernière modification
            latest_log = sorted(
                log_files,
                key=lambda f: f.stat().st_mtime,
                reverse=True
            )[0]
            log_analysis = analyze_log_file(str(latest_log))
        else:
            # Dossier existe mais vide → démonstration
            log_analysis = analyze_log_string(
                "[2026-10-01 02:41:23] INFO Pipeline started\n"
                "[2026-10-01 02:41:45] ERROR OutOfMemoryError: Java heap space\n"
                "[2026-10-01 02:42:01] ERROR Connection refused to PostgreSQL\n"
                "[2026-10-01 02:43:00] CRITICAL Pipeline failed after 3 retries"
            )
    else:
        # Dossier ./logs/ inexistant → démonstration
        log_analysis = analyze_log_string(
            "[2026-10-01 02:41:23] INFO Pipeline started\n"
            "[2026-10-01 02:41:45] ERROR OutOfMemoryError: Java heap space\n"
            "[2026-10-01 02:42:01] ERROR Connection refused to PostgreSQL\n"
            "[2026-10-01 02:43:00] CRITICAL Pipeline failed after 3 retries"
        )

    # Construire le prompt avec le résumé structuré des logs
    # On envoie le résumé, pas les 1000 lignes brutes
    prompt = f"""You are a Data Engineering assistant analyzing pipeline logs.
Based on the log analysis below, answer the user's question.

LOG SUMMARY:
{log_analysis['summary']}

ERRORS FOUND ({log_analysis['error_count']}):
{chr(10).join([f"- [{e['timestamp']}] {e['message']}" for e in log_analysis['recent_errors']])}

CRITICAL ISSUES ({log_analysis['critical_count']}):
{chr(10).join([f"- [{e['timestamp']}] {e['message']}" for e in log_analysis['criticals']])}

TOP ERROR PATTERNS:
{chr(10).join([f"- '{p[0]}' appeared {p[1]} times" for p in log_analysis['top_error_patterns']])}

QUESTION: {question}

ANSWER:"""

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"{settings.OLLAMA_BASE_URL}/api/generate",
            json={
                "model"  : settings.LLM_MODEL,
                "prompt" : prompt,
                "stream" : False,
                "options": {"temperature": 0.1, "num_predict": 512}
            }
        )
        response.raise_for_status()
        answer = response.json()["response"]

    return {
        "log_analysis": log_analysis,
        "answer"      : answer,
        "sources"     : [],
        "steps"       : state["steps"] + ["log_analyzer"]
    }