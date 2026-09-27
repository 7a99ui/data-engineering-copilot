from fastapi import FastAPI
from pydantic import BaseModel
from app.llm.client import ollama_client # on importe l'instance unique du client Ollama qu'on a créée dans client.py

app = FastAPI(title="Data Engineering Copilot", version="0.1.0") # On crée l'application FastAPI. C'est l'objet central — tout se branche dessus (routes, middleware...)

class ChatRequest(BaseModel): # Ou définit la forme de ce que l'API attend. Quand quelqu'un appelle /chat, il doit envoyer un JSON avec un champ question de type str
    question: str

class ChatResponse(BaseModel): # On définit la forme de ce que l'API va renvoyer. La réponse contiendra toujours deux champs : le texte généré par le LLM et le nom du modèle utilisé
    answer: str
    model: str

@app.get("/health") 
# Cette route ne fait rien de complexe — elle répond juste {"status": "ok"} pour dire "l'API est vivante"
# C'est utilisé par Docker, les outils de monitoring, ou moi-même pour vérifier que le serveur tourne.
async def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse) 
async def chat(request: ChatRequest): # ← reçoit un ChatRequest rempli
    answer = await ollama_client.chat(request.question) # ← appelle client.py et extrait la question du ChatRequest
    return ChatResponse( # ← construit un ChatResponse
        answer=answer, # ← avec la réponse d'Ollama
        model=ollama_client.model
    )