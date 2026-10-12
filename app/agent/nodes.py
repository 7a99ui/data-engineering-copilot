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
from app.agent.tools.sql_utils import extract_sql

# ── Imports Python standard ──
import asyncio
import time                               # mesurer les durées des nœuds
from pathlib import Path
import httpx
from app.config import settings

# ── Import LangFuse — Phase 7b ──
# is_enabled() retourne False si les clés ne sont pas configurées
# → LangFuse désactivé silencieusement, le reste fonctionne normalement
#
# IMPORTANT : l'état LangGraph est sauvegardé par le checkpointer (msgpack).
# L'objet "trace" de LangFuse n'est PAS sérialisable → on ne stocke dans
# l'état que son ID (un simple texte), et on reconstruit l'objet avec
# get_trace(state) dans chaque nœud.
from app.observability import langfuse, is_enabled, new_trace, get_trace


async def analyze_intent_node(state: dict) -> dict:
    """
    Nœud 1 — Analyse l'intent de la question.
    Classifie en 6 catégories : rag_search / direct_answer / chitchat
                                / db_inspect / sql_query / log_analysis
    """
    question = state["question"]

    # ── LangFuse : créer la trace pour cette requête complète ──
    # Une trace = une requête utilisateur de bout en bout.
    # new_trace() retourne (id, objet) ou (None, None) si LangFuse est désactivé.
    trace_id, trace = new_trace(
        name    ="agent_request",
        input   =question,
        metadata={"mode": "agent"}
    )

    # Mesurer la durée
    t0 = time.time()

    # Classifier la question avec le LLM (temperature=0.0 → déterministe)
    intent = await classify_intent(question)

    duration = time.time() - t0

    # ── LangFuse : span pour ce nœud ──
    if trace:
        trace.span(
            name    ="analyze_intent",
            input   ={"question": question},
            output  ={"intent": intent},
            metadata={"duration_seconds": round(duration, 3)}
        )

    # On retourne UNIQUEMENT les champs mis à jour
    # LangGraph fusionne avec l'état existant automatiquement
    return {
        "intent"   : intent,
        "steps"    : state["steps"] + ["analyze_intent"],
        "_lf_trace": trace_id    # l'ID (texte), jamais l'objet trace
    }


async def rag_search_node(state: dict) -> dict:
    """
    Nœud 2a — Recherche RAG hybride Sprint 4.
    Utilisé quand intent == "rag_search".
    Dense + BM25 + RRF + CrossEncoder + génération.
    """
    question = state["question"]
    trace    = get_trace(state)   # reconstruit la trace depuis l'ID de l'état

    t0 = time.time()

    # Retrieval hybride → top-6 chunks fusionnés
    chunks = hybrid_retrieve(question, k=6)

    # CrossEncoder reranking → top-3 chunks les plus pertinents
    reranked_chunks = await rerank(question, chunks, top_k=3)

    # Génération avec le prompt RAG structuré
    answer = await generate(question, reranked_chunks)

    duration = time.time() - t0

    # Construire les sources à retourner à l'utilisateur
    sources = []
    for chunk in reranked_chunks:
        sources.append({
            "source": chunk["payload"]["source"],
            "page"  : chunk["payload"].get("page", 1),
            "score" : round(chunk["score"], 2),
            "chunk" : chunk["payload"]["text"][:150] + "..."
        })

    # ── LangFuse : span avec métriques RAG ──
    if trace:
        trace.span(
            name    ="rag_search",
            input   ={"question": question},
            output  ={
                "answer"      : answer[:500],
                "chunks_count": len(reranked_chunks),
                "top_score"   : float(reranked_chunks[0]["score"]) if reranked_chunks else 0,
                "sources"     : [s["source"] for s in sources]
            },
            metadata={"duration_seconds": round(duration, 3)}
        )
        trace.update(output=answer[:500])

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
    trace    = get_trace(state)

    prompt = f"""You are an expert Data Engineering assistant.
Answer the following general Data Engineering question clearly and concisely.
Use your knowledge to provide a helpful, accurate answer.

Question: {question}

Answer:"""

    t0 = time.time()

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

    duration = time.time() - t0

    # ── LangFuse : span + generation ──
    if trace:
        trace.span(
            name    ="direct_answer",
            input   ={"question": question},
            output  ={"answer": answer[:500]},
            metadata={"duration_seconds": round(duration, 3)}
        )
        trace.generation(
            name    ="direct_answer_llm",
            model   =settings.LLM_MODEL,
            input   =prompt[:2000],
            output  =answer[:2000],
            metadata={"duration_seconds": round(duration, 3)}
        )
        trace.update(output=answer[:500])

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
    trace    = get_trace(state)

    prompt = f"""You are a friendly Data Engineering assistant.
Respond briefly and naturally to this message.
If it's a greeting, greet back. If it's off-topic, politely redirect
to Data Engineering topics.

Message: {question}

Response:"""

    t0 = time.time()

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

    duration = time.time() - t0

    # ── LangFuse : span chitchat ──
    if trace:
        trace.span(
            name    ="chitchat",
            input   ={"question": question},
            output  ={"answer": answer[:300]},
            metadata={"duration_seconds": round(duration, 3)}
        )
        trace.update(output=answer[:300])

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
    trace = get_trace(state)

    # ── LangFuse : finaliser la trace complète ──
    if trace:
        trace.span(
            name    ="format_response",
            input   ={},
            output  ={"steps": state["steps"]},
            metadata={"total_steps": len(state["steps"])}
        )
        trace.update(
            metadata={
                "intent"     : state.get("intent"),
                "total_steps": len(state["steps"]) + 1,
                "has_sources": bool(state.get("sources"))
            }
        )
        # flush() est bloquant → on l'exécute dans un thread
        # pour ne pas figer le serveur pendant l'envoi à LangFuse
        await asyncio.to_thread(langfuse.flush)

    return {
        "steps"    : state["steps"] + ["format_response"],
        "_lf_trace": None   # trace terminée → on nettoie l'état
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
    trace    = get_trace(state)

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

    t0 = time.time()

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

    duration = time.time() - t0

    # ── LangFuse : span db_inspect ──
    if trace:
        trace.span(
            name    ="db_inspect",
            input   ={"question": question},
            output  ={
                "answer"      : answer[:500],
                "tables_found": db_info["tables_count"]
            },
            metadata={"duration_seconds": round(duration, 3)}
        )
        trace.update(output=answer[:500])

    return {
        "db_info": db_info,
        "answer" : answer,
        "sources": [],
        "steps"  : state["steps"] + ["db_inspect"]
    }


async def sql_query_node(state: dict) -> dict:
    """
    Nœud SQL Query — génère la requête SQL depuis la question.
    C'est le nœud AVANT l'interruption Human in the Loop.

    - extract_sql() extrait la requête du texte du LLM
    - l'historique conversationnel permet de comprendre "now only the failed ones"
    - validate_query() refuse tout ce qui n'est pas un SELECT sûr
    """
    question = state["question"]
    trace    = get_trace(state)

    # Récupérer la structure DB pour que le LLM génère du SQL correct
    db_info = inspect_database()
    tables_summary = "\n".join([
        f"- {t['table_name']}: " +
        ", ".join([c['column_name'] for c in t['columns']])
        for t in db_info["tables"]
    ])

    # ── Historique conversationnel ──
    # "or []" : la clé peut exister avec la valeur None (Optional[list])
    history = state.get("history") or []
    history_text = "\n".join(
        f"{m['role']}: {m['content'][:500]}"
        for m in history[-6:]   # 6 derniers messages max
    )

    # ── Prompt ──
    # Ne pas terminer par une étiquette du genre "SQL Query (SELECT only):"
    # → le LLM la recopiait dans sa réponse. On termine par "SQL:".
    sql_prompt = f"""You are a PostgreSQL expert. Write ONE read-only SQL query.
Return ONLY the SQL query. No explanation, no markdown, no labels.

Available tables and columns:
{tables_summary}

Conversation so far (context only):
{history_text or '(none)'}

Question: {question}

Rules:
- Write a NEW query that answers the Question. Do NOT repeat a previous query unless the Question asks for the same data.
- Use the conversation only to resolve references such as "those", "only the failed ones", "same thing for ...".
- If the user asks to fix or correct the query, correct the last SQL query from the conversation.
- Only SELECT queries are allowed. If the user asks to modify, delete or insert data, output exactly: SELECT 'read-only access' AS message
- To count tables use: SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'

SQL:"""

    t0 = time.time()

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"{settings.OLLAMA_BASE_URL}/api/generate",
            json={
                "model"  : settings.LLM_MODEL,
                "prompt" : sql_prompt,
                "stream" : False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 300   # assez pour les requêtes avec jointure
                }
            }
        )
        response.raise_for_status()

    duration = time.time() - t0

    # ── Extraction robuste de la requête ──
    # Gère : texte avant/après, backticks, double instruction, ';' final
    sql_query = extract_sql(response.json()["response"])

    # Si extract_sql retourne vide → le LLM n'a pas généré de SELECT
    if not sql_query:
        if trace:
            trace.span(
                name    ="sql_query",
                input   ={"question": question},
                output  ={"error": "no_sql_generated"},
                metadata={"duration_seconds": round(duration, 3)}
            )
        return {
            "sql_query"   : "",
            "sql_approved": False,
            "answer"      : "I could not generate a valid SQL query. Please rephrase your question.",
            "sources"     : [],
            "steps"       : state["steps"] + ["sql_query", "sql_rejected"]
        }

    # Valider la sécurité AVANT de demander confirmation
    is_safe, error_msg = validate_query(sql_query)

    # ── LangFuse : span sql_query ──
    if trace:
        trace.span(
            name    ="sql_query",
            input   ={"question": question},
            output  ={
                "sql_query": sql_query,
                "is_safe"  : is_safe,
                "error"    : error_msg if not is_safe else None
            },
            metadata={"duration_seconds": round(duration, 3)}
        )

    if not is_safe:
        # Requête dangereuse → refus immédiat sans Human in the Loop
        return {
            "sql_query"   : sql_query,
            "sql_approved": False,
            "answer"      : f"I cannot execute this query for security reasons: {error_msg}",
            "sources"     : [],
            "steps"       : state["steps"] + ["sql_query", "sql_rejected"]
        }

    # Requête safe → on attend la confirmation humaine (interrupt_before=["sql_execute"])
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

    Étapes :
    1. Vérifie que sql_approved est bien True
    2. Exécute la requête via execute_query()
    3. Formule la réponse :
       - 0 ligne        → message fixe
       - plusieurs lignes → message fixe, le tableau de l'interface fait foi
       - 1 ligne        → le LLM la résume en une ou deux phrases
    """
    sql_query    = state["sql_query"]
    sql_approved = state.get("sql_approved")
    trace        = get_trace(state)

    # Sécurité — ne s'exécute que si explicitement approuvé
    if not sql_approved:
        return {
            "answer": "SQL execution was not approved.",
            "steps" : state["steps"] + ["sql_execute_denied"]
        }

    # Exécuter la requête validée et approuvée
    t0          = time.time()
    result      = execute_query(sql_query)
    db_duration = time.time() - t0

    if not result["success"]:
        # Erreur PostgreSQL — syntaxe incorrecte, table inexistante...
        if trace:
            trace.span(
                name    ="sql_execute",
                input   ={"sql_query": sql_query},
                output  ={"error": result["error"]},
                metadata={"duration_seconds": round(db_duration, 3), "success": False}
            )
        return {
            "sql_result": result,
            "answer"    : f"SQL execution failed: {result['error']}",
            "sources"   : [],
            "steps"     : state["steps"] + ["sql_execute_error"]
        }

    row_count    = result["row_count"]
    llm_duration = 0.0

    if row_count == 0:
        answer = "The query ran successfully but returned no rows."

    elif row_count > 1:
        # Plusieurs lignes : pas de résumé LLM (il réordonne et déforme les valeurs).
        # L'interface affiche le tableau complet avec st.dataframe.
        answer = f"The query returned {row_count} rows. See the table below."

    else:
        # Une seule ligne (ex : COUNT) → une phrase suffit
        answer_prompt = f"""Based on this SQL query result, answer the user's question in one or two sentences.

Query executed: {sql_query}
Result: {result['rows'][0]}

Original question: {state['question']}

Answer:"""

        t1 = time.time()
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{settings.OLLAMA_BASE_URL}/api/generate",
                json={
                    "model"  : settings.LLM_MODEL,
                    "prompt" : answer_prompt,
                    "stream" : False,
                    "options": {"temperature": 0.1, "num_predict": 200}
                }
            )
            response.raise_for_status()
            answer = response.json()["response"]
        llm_duration = time.time() - t1

    # ── LangFuse : span sql_execute ──
    if trace:
        trace.span(
            name    ="sql_execute",
            input   ={"sql_query": sql_query},
            output  ={
                "row_count": row_count,
                "answer"   : answer[:300]
            },
            metadata={
                "db_duration_s" : round(db_duration, 3),
                "llm_duration_s": round(llm_duration, 3),
                "success"       : True
            }
        )
        trace.update(output=answer[:500])

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
    trace    = get_trace(state)

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

    t0 = time.time()

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

    duration = time.time() - t0

    # ── LangFuse : span log_analyzer ──
    if trace:
        trace.span(
            name    ="log_analyzer",
            input   ={"question": question},
            output  ={
                "answer"      : answer[:500],
                "errors_found": log_analysis["error_count"],
                "criticals"   : log_analysis["critical_count"]
            },
            metadata={"duration_seconds": round(duration, 3)}
        )
        trace.update(output=answer[:500])

    return {
        "log_analysis": log_analysis,
        "answer"      : answer,
        "sources"     : [],
        "steps"       : state["steps"] + ["log_analyzer"]
    }