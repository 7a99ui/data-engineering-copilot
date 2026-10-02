# FastAPI — le framework qui crée les routes de l'API
from fastapi import FastAPI

# BaseModel — la classe de base Pydantic pour valider les données
from pydantic import BaseModel

# Le client Ollama — instance unique créée dans client.py (pattern Singleton)
# utilisé pour la route /chat (Sprint 1)
from app.llm.client import ollama_client

# Le pipeline RAG complet — orchestrate retriever + generator
# utilisé pour la route /ask (Sprint 3 et 4)
from app.rag.pipeline import rag_pipeline

# Le graphe LangGraph — instance unique compilée au démarrage
# utilisé pour la nouvelle route /agent (Sprint 5)
from app.agent.graph import agent_graph

# ── Création de l'application ──
# on passe à 0.3.0 car on ajoute l'agent LangGraph — fonctionnalité majeure
app = FastAPI(title="Data Engineering Copilot", version="0.3.0")


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
# MODÈLES DE DONNÉES — Sprint 5 (/agent)
# ════════════════════════════════════════

class AgentRequest(BaseModel):
    # Ce que l'API attend pour /agent
    # un seul champ — l'agent décide lui-même comment répondre
    question: str

class AgentResponse(BaseModel):
    # Ce que l'API retourne pour /agent
    # answer  : la réponse générée (RAG ou directe selon l'intent)
    # intent  : l'intent classifié par l'agent
    #           "rag_search" | "direct_answer" | "chitchat"
    # sources : les chunks utilisés (vide si pas de RAG)
    # steps   : le chemin parcouru dans le graphe LangGraph
    #           ex: ["analyze_intent", "rag_search", "format_response"]
    #           utile pour le debug et la traçabilité
    # model   : le nom du modèle LLM utilisé
    answer : str
    intent : str
    sources: list[SourceItem]
    steps  : list[str]
    model  : str


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


# ── Route /ask — Sprint 3 & 4 (RAG) ──
@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    # Pipeline RAG fixe — toujours le même chemin :
    # retrieve → augment → generate
    # mode "hybrid" → Dense + BM25 + RRF + CrossEncoder (Sprint 4)
    # mode "dense"  → Dense seul (Sprint 3) pour comparaison

    result = await rag_pipeline(
        question=request.question,
        k       =request.k,
        mode    =request.mode
    )

    # SourceItem(**s) déverse chaque dict source dans un objet Pydantic
    # ex: {"source": "spark.md", "page": 1, "score": 0.94, "chunk": "..."}
    #     devient SourceItem(source="spark.md", page=1, score=0.94, chunk="...")
    return AskResponse(
        answer =result["answer"],
        sources=[SourceItem(**s) for s in result["sources"]],
        model  =result["model"],
        mode   =result["mode"]
    )


# ── Route /agent — Sprint 5 (LangGraph) ──
@app.post("/agent", response_model=AgentResponse)
async def agent(request: AgentRequest):
    # Agent LangGraph — pipeline intelligent qui s'adapte à la question
    # Différence clé vs /ask :
    # /ask  → toujours RAG, toujours le même chemin
    # /agent → analyse l'intent d'abord, choisit la meilleure stratégie

    # ── État initial du graphe ──
    # On fournit uniquement ce qu'on sait au départ
    # Les autres champs (intent, chunks, answer, sources)
    # seront remplis par les nœuds au fil de l'exécution
    initial_state = {
        "question": request.question,
        "intent"  : None,   # ← classifié par analyze_intent_node
        "chunks"  : None,   # ← rempli par rag_search_node si besoin
        "answer"  : None,   # ← rempli par le nœud actif
        "sources" : [],     # ← rempli par rag_search_node si besoin
        "steps"   : []      # ← mis à jour à chaque nœud parcouru
    }

    # ── Invoquer le graphe ──
    # ainvoke() = version asynchrone de invoke()
    # exécute tous les nœuds dans l'ordre défini dans graph.py :
    # START → analyze_intent → [rag_search | direct_answer | chitchat]
    #       → format_response → END
    # retourne l'état FINAL après tous les nœuds
    final_state = await agent_graph.ainvoke(initial_state)

    # ── Construire les sources ──
    # final_state["sources"] peut être None si pas de RAG
    # on utilise "or []" pour éviter l'erreur si None
    sources = [
        SourceItem(**s)
        for s in (final_state.get("sources") or [])
    ]

    # ── Retourner la réponse enrichie ──
    # on expose intent et steps pour la transparence et le debug
    # un recruteur peut voir exactement comment l'agent a raisonné
    return AgentResponse(
        answer =final_state["answer"],
        intent =final_state["intent"],
        sources=sources,
        steps  =final_state["steps"],
        model  ="qwen2.5:3b"
    )