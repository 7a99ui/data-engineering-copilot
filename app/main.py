# FastAPI — le framework qui crée les routes de l'API
from fastapi import FastAPI

# BaseModel — la classe de base Pydantic pour valider les données
from pydantic import BaseModel

# Le client Ollama — instance unique créée dans client.py (pattern Singleton)
# utilisé pour la route /chat (Sprint 1)
from app.llm.client import ollama_client

# Le pipeline RAG complet — orchestrate retriever + generator
# utilisé pour la nouvelle route /ask (Sprint 3)
from app.rag.pipeline import rag_pipeline

# ── Création de l'application ──
# FastAPI() crée l'objet central de l'API
# title et version apparaissent dans la page /docs (Swagger)
# on passe de 0.1.0 à 0.2.0 car on ajoute une fonctionnalité majeure (RAG)
app = FastAPI(title="Data Engineering Copilot", version="0.2.0")


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
    model: str


# ════════════════════════════════════════
# MODÈLES DE DONNÉES — Sprint 3 (/ask)
# ════════════════════════════════════════

class AskRequest(BaseModel):
    # Ce que l'API attend pour /ask
    # question : la question de l'utilisateur en anglais
    # k : combien de chunks récupérer dans Qdrant (défaut 3)
    #     valeur par défaut = 3 → optionnel dans la requête JSON
    question: str
    k: int = 3
    # mode de retrieval — "hybrid" par défaut (Sprint 4)
    # "dense" → Sprint 3 pour comparaison
    # Literal["hybrid", "dense"] → Pydantic accepte SEULEMENT ces deux valeurs
    mode: str = "hybrid"

class SourceItem(BaseModel):
    # Représente UNE source utilisée pour répondre
    # retournée dans la liste "sources" de AskResponse
    # source      : nom du fichier (ex: "spark.md")
    # page        : numéro de page dans le document
    # score       : score de similarité cosine (0.0 à 1.0)
    # chunk       : extrait du texte utilisé (150 premiers caractères)
    source: str
    page  : int
    score : float
    chunk : str

class AskResponse(BaseModel):
    # Ce que l'API retourne pour /ask
    # answer  : la réponse générée par le LLM à partir des documents
    # sources : liste des chunks utilisés pour répondre (avec scores)
    # model   : le nom du modèle LLM utilisé
    answer : str
    sources: list[SourceItem]
    model  : str
    # retourner le mode utilisé pour transparence
    mode: str


# ════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════

# ── Route de santé ──
@app.get("/health")
async def health():
    # Route la plus simple — vérifie que l'API tourne
    # utilisée par Docker, outils de monitoring, ou manuellement
    # ne nécessite aucun paramètre, retourne toujours {"status": "ok"}
    return {"status": "ok"}


# ── Route /chat — Sprint 1 ──
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # route du Sprint 1 — LLM seul sans RAG
    # répond avec sa culture générale, pas avec tes documents
    # toujours utile pour des questions générales

    # request.question : le texte extrait du JSON entrant par Pydantic
    # ollama_client.chat() : appelle Ollama via HTTP (voir client.py)
    # await : attend la réponse sans bloquer le serveur
    answer = await ollama_client.chat(request.question)

    # on retourne un ChatResponse — Pydantic valide que answer et model
    # sont bien des strings avant d'envoyer la réponse JSON
    return ChatResponse(
        answer=answer,
        model=ollama_client.model
    )


# ── Route /ask — Sprint 3 (RAG) ──
@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    # route principale du Sprint 3 — LLM + RAG
    # répond à partir de TES documents stockés dans Qdrant
    # retourne la réponse + les sources utilisées + les scores

    # rag_pipeline() orchestre les 3 étapes :
    # 1. retrieve  → vectorise la question, cherche dans Qdrant
    # 2. augment   → construit le prompt avec le contexte trouvé
    # 3. generate  → appelle le LLM avec le prompt enrichi
    # request.question : la question de l'utilisateur
    # request.k        : nombre de chunks à récupérer (défaut 3)
    result = await rag_pipeline(
        question=request.question,
        k=request.k,
        mode=request.mode      # ← nouveau paramètre
    )

    # result est un dict avec 3 clés : answer, sources, model
    # on construit un AskResponse — Pydantic valide chaque champ
    # SourceItem(**s) déverse chaque dict source dans un objet SourceItem
    # ex: {"source": "spark.md", "page": 1, "score": 0.94, "chunk": "..."}
    #     devient SourceItem(source="spark.md", page=1, score=0.94, chunk="...")
    return AskResponse(
        answer =result["answer"],
        sources=[SourceItem(**s) for s in result["sources"]],
        model  =result["model"],
        mode   =result["mode"]  # ← nouveau champ
    )