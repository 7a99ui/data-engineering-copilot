import httpx # importation la bibliothèque qui permet de faire des requêtes HTTP. C'est mon "téléphone" pour appeler Ollama.
from app.config import settings # importation l'objet settings de mon config.py — celui qui contient OLLAMA_BASE_URL, LLM_MODEL, etc

class OllamaClient: # une boîte qui va regrouper tout ce qui concerne la communication avec Ollama
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.LLM_MODEL

    async def chat(self, prompt: str) -> str: #le paramètre doit être une chaîne de texte, cette fonction promet de retourner une chaîne de texte
        async with httpx.AsyncClient(timeout=120) as client:  # si Ollama ne répond pas en 120 secondes, on abandonne et on lève une erreur
            response = await client.post( # attend le résultat avant de continuer
                f"{self.base_url}/api/generate", # construit l'URL complète : http://localhost:11434/api/generate. C'est l'endpoint qu'Ollama expose pour générer du texte.
                json={ # C'est le corps de la requête envoyée à Ollama
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False # envoie toute la réponse d'un coup" au lieu de l'envoyer mot par mot
                }
            )
            response.raise_for_status() # Si Ollama répond avec une erreur (code 404, 500...), cette ligne lève automatiquement une exception Python claire.
            return response.json()["response"] # Ollama renvoie un JSON, on extrait uniquement le texte de la réponse avec ["response"] et le retournes.

ollama_client = OllamaClient() # On crée une seule instance du client au démarrage du programme