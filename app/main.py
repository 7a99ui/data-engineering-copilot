# FastAPI — le framework qui crée les routes de l'API
from fastapi import FastAPI

# BaseModel — la classe de base Pydantic pour valider les données
from pydantic import BaseModel

# Optional — pour les champs optionnels dans les modèles Pydantic
# utilisé dans AgentRequest (thread_id, sql_approved optionnels)
from typing import Optional

# uuid — génère des identifiants uniques pour les threads de conversation
# chaque conversation a son propre thread_id pour le checkpointer LangGraph
import uuid

# Le client Ollama — instance unique créée dans client.py (pattern Singleton)
# utilisé pour la route /chat (Sprint 1)
from app.llm.client import ollama_client

# Le pipeline RAG complet — orchestrate retriever + generator
# utilisé pour la route /ask (Sprint 3 et 4)
from app.rag.pipeline import rag_pipeline

# Le graphe LangGraph — instance unique compilée au démarrage
# utilisé pour la route /agent (Sprint 5 et 6)
# Sprint 6 : graphe étendu avec DB inspector, SQL executor, Log analyzer
#            + Human in the Loop via interrupt_before=["sql_execute"]
from app.agent.graph import agent_graph

# ── Création de l'application ──
# on passe à 0.4.0 car on ajoute les outils spécialisés et HITL — Sprint 6
app = FastAPI(title="Data Engineering Copilot", version="0.4.0")


# ════════════════════════════════════════
# MODÈLES DE DONNÉES — Sprint 1 (/chat)
# ════════════════════════════════════════

class ChatRequest(BaseModel):
    # Ce que l'API attend pour /chat
    # un seul champ obligatoire : la question en texte
    question: str

class ChatResponse(BaseModel):
    # Ce que l'API retourne pour /chat
    # la réponse du LLM + le nom du modèle utilisé
    answer: str
    model : str


# ════════════════════════════════════════
# MODÈLES DE DONNÉES — Sprint 3 & 4 (/ask)
# ════════════════════════════════════════

class AskRequest(BaseModel):
    # Ce que l'API attend pour /ask
    # question : la question de l'utilisateur en anglais
    # k        : combien de chunks récupérer dans Qdrant (défaut 3)
    # mode     : "hybrid" (Sprint 4) ou "dense" (Sprint 3)
    question: str
    k   : int = 3
    mode: str = "hybrid"

class SourceItem(BaseModel):
    # Représente UNE source utilisée pour répondre
    # utilisée dans AskResponse ET AgentResponse
    # source : nom du fichier (ex: "spark.md")
    # page   : numéro de page dans le document
    # score  : score de pertinence (cosine ou CrossEncoder)
    # chunk  : extrait des 150 premiers caractères du chunk
    source: str
    page  : int
    score : float
    chunk : str

class AskResponse(BaseModel):
    # Ce que l'API retourne pour /ask
    # answer  : la réponse générée par le LLM à partir des documents
    # sources : liste des chunks utilisés avec leurs scores
    # model   : le nom du modèle LLM utilisé
    # mode    : la stratégie utilisée ("hybrid" ou "dense")
    answer : str
    sources: list[SourceItem]
    model  : str
    mode   : str


# ════════════════════════════════════════
# MODÈLES DE DONNÉES — Sprint 5 & 6 (/agent)
# ════════════════════════════════════════

class AgentRequest(BaseModel):
    # Ce que l'API attend pour /agent
    #
    # ── Nouvelle question (usage normal) ──
    # question     : la question de l'utilisateur
    # thread_id    : None → généré automatiquement
    # sql_approved : None → pas de décision SQL en cours
    #
    # ── Reprise après confirmation SQL (Human in the Loop) ──
    # question     : peut être omise (None)
    # thread_id    : OBLIGATOIRE → identifie la conversation à reprendre
    # sql_approved : True (approuver) ou False (refuser) l'exécution SQL

    # La question posée à l'agent
    # Optional car lors d'une reprise SQL, on n'en a pas besoin
    question: Optional[str] = None

    # Identifiant unique de la conversation
    # None = nouvelle conversation → un ID sera généré automatiquement
    # Valeur = reprise d'une conversation existante (après interruption SQL)
    thread_id: Optional[str] = None

    # Décision humaine sur l'exécution SQL
    # None    = nouvelle question, pas de SQL en attente
    # True    = approuver l'exécution de la requête SQL
    # False   = refuser l'exécution de la requête SQL
    sql_approved: Optional[bool] = None


class AgentResponse(BaseModel):
    # Ce que l'API retourne pour /agent
    #
    # answer           : la réponse générée
    # intent           : l'intent classifié (rag_search, db_inspect, sql_query...)
    # sources          : chunks RAG utilisés (vide si pas de RAG)
    # steps            : chemin parcouru dans le graphe LangGraph
    # model            : nom du modèle LLM utilisé
    # thread_id        : ID de la conversation — à conserver pour la reprise SQL
    # sql_query        : la requête SQL générée (si en attente de confirmation)
    # pending_approval : True si on attend une confirmation SQL humaine
    answer          : str
    intent          : str
    sources         : list[SourceItem]
    steps           : list[str]
    model           : str

    # thread_id retourné pour que le client puisse reprendre la conversation
    # le client doit stocker cet ID et le renvoyer pour approuver/refuser le SQL
    thread_id       : str

    # sql_query retourné si en attente de confirmation
    # le client affiche cette requête à l'humain pour validation
    # None si pas de SQL en attente
    sql_query       : Optional[str] = None

    # True si le graphe s'est interrompu et attend une confirmation SQL
    # False dans tous les autres cas
    pending_approval: bool = False


# ════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════

# ── Route de santé ──
@app.get("/health")
async def health():
    # Route la plus simple — vérifie que l'API tourne
    # utilisée par Docker, outils de monitoring, ou manuellement
    # retourne toujours {"status": "ok"}
    return {"status": "ok"}


# ── Route /chat — Sprint 1 ──
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # LLM seul sans RAG — répond avec sa culture générale
    # toujours utile pour des questions générales non liées aux docs
    # ollama_client.chat() appelle Ollama via HTTP (voir client.py)
    # await attend la réponse sans bloquer le serveur
    answer = await ollama_client.chat(request.question)

    return ChatResponse(
        answer=answer,
        model =ollama_client.model
    )


# ── Route /ask — Sprint 3 & 4 (RAG fixe) ──
@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    # Pipeline RAG fixe — toujours le même chemin
    # mode "hybrid" → Dense + BM25 + RRF + CrossEncoder (Sprint 4)
    # mode "dense"  → Dense seul (Sprint 3) pour comparaison

    result = await rag_pipeline(
        question=request.question,
        k       =request.k,
        mode    =request.mode
    )

    # SourceItem(**s) déverse chaque dict source dans un objet Pydantic
    return AskResponse(
        answer =result["answer"],
        sources=[SourceItem(**s) for s in result["sources"]],
        model  =result["model"],
        mode   =result["mode"]
    )


# ── Route /agent — Sprint 5 & 6 (LangGraph + outils + HITL) ──
@app.post("/agent", response_model=AgentResponse)
async def agent(request: AgentRequest):
    """
    Route agent principale — deux usages possibles :

    ── Usage 1 : Nouvelle question ──
    POST /agent
    {"question": "What tables are in the database ?"}
    → L'agent analyse l'intent et choisit l'outil adapté
    → Si SQL → retourne pending_approval=True + sql_query + thread_id

    ── Usage 2 : Reprise après confirmation SQL ──
    POST /agent
    {"thread_id": "abc-123", "sql_approved": true}
    → Le graphe reprend depuis sql_execute avec l'approbation
    → Retourne les résultats SQL + réponse LLM
    """

    # ── Générer ou réutiliser le thread_id ──
    # thread_id identifie la conversation dans le checkpointer (MemorySaver)
    # chaque conversation a sa propre mémoire sauvegardée
    # str(uuid.uuid4()) génère un ID unique comme "f47ac10b-58cc-4372-a567-..."
    thread_id = request.thread_id or str(uuid.uuid4())

    # Configuration LangGraph — passée à ainvoke() pour identifier la conversation
    # le checkpointer utilise thread_id pour retrouver l'état sauvegardé
    config = {"configurable": {"thread_id": thread_id}}

    # ── Cas 1 : Reprise après décision SQL ──
    # thread_id fourni + sql_approved fourni (True ou False)
    # → l'humain a pris une décision sur la requête SQL en attente
    if request.thread_id and request.sql_approved is not None:

        # Mettre à jour l'état avec la décision humaine
        # aupdate_state() modifie l'état sauvegardé dans le checkpointer
        # sans relancer tout le graphe depuis le début
        await agent_graph.aupdate_state(
            config,
            # On met à jour uniquement sql_approved dans l'état
            {"sql_approved": request.sql_approved}
        )

        # Reprendre le graphe depuis le point d'interruption
        # ainvoke(None, config) = "reprends où tu t'étais arrêté"
        # None = pas de nouvel état initial — on reprend l'état sauvegardé
        final_state = await agent_graph.ainvoke(None, config)

    # ── Cas 2 : Nouvelle question ──
    else:
        # Vérification — une question est obligatoire pour une nouvelle conversation
        if not request.question:
            return AgentResponse(
                answer          ="Please provide a question.",
                intent          ="unknown",
                sources         =[],
                steps           =[],
                model           ="qwen2.5:3b",
                thread_id       =thread_id,
                sql_query       =None,
                pending_approval=False
            )

        # État initial complet avec tous les champs du Sprint 6
        # Les champs None/[] seront remplis par les nœuds au fil de l'exécution
        initial_state = {
            "question"    : request.question,
            "intent"      : None,   # ← classifié par analyze_intent_node
            "chunks"      : None,   # ← rempli par rag_search_node si besoin
            "answer"      : None,   # ← rempli par le nœud actif
            "sources"     : [],     # ← rempli par rag_search_node si besoin
            "steps"       : [],     # ← mis à jour à chaque nœud parcouru
            "sql_query"   : None,   # ← rempli par sql_query_node si besoin
            "sql_approved": None,   # ← None = en attente, True/False = décidé
            "sql_result"  : None,   # ← rempli par sql_execute_node si besoin
            "log_analysis": None,   # ← rempli par log_analyzer_node si besoin
            "db_info"     : None,   # ← rempli par db_inspect_node si besoin
        }

        # Invoquer le graphe avec l'état initial et la config de thread
        # ainvoke() exécute les nœuds jusqu'à END ou jusqu'à une interruption
        # retourne l'état FINAL (ou l'état au moment de l'interruption)
        final_state = await agent_graph.ainvoke(initial_state, config)

    # ── Détecter si on est en attente de confirmation SQL ──
    # pending = True si :
    # - une requête SQL a été générée (sql_query rempli)
    # - ET elle n'a pas encore été approuvée (sql_approved est None)
    # Dans ce cas le graphe s'est interrompu — on attend la décision humaine
    pending = (
        final_state.get("sql_query") is not None and
        final_state.get("sql_approved") is None
    )

    # ── Construire les sources ──
    # final_state["sources"] peut être None si pas de RAG
    # "or []" évite une erreur si None
    sources = [
        SourceItem(**s)
        for s in (final_state.get("sources") or [])
    ]

    # ── Retourner la réponse complète ──
    return AgentResponse(
        answer          =final_state.get("answer", ""),
        intent          =final_state.get("intent", "unknown"),
        sources         =sources,
        steps           =final_state.get("steps", []),
        model           ="qwen2.5:3b",

        # thread_id retourné pour que le client puisse reprendre après SQL
        # le client stocke cet ID et le renvoie avec sql_approved
        thread_id       =thread_id,

        # sql_query retourné si en attente de confirmation
        # le client affiche cette requête à l'humain
        sql_query       =final_state.get("sql_query"),

        # True si le graphe attend une confirmation SQL humaine
        pending_approval=pending
    )