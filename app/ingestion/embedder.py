import httpx
from app.config import settings
# Le nom du modèle d'embeddings
EMBEDDING_MODEL = "nomic-embed-text"

# Elle prend une liste de textes et retourne une liste de listes de nombres. Chaque texte donne une liste de 768 nombres.
def embed_texts(texts: list[str]) -> list[list[float]]:
    embeddings = [] # Liste vide qu'on va remplir — un vecteur par texte.

    for i, text in enumerate(texts): # enumerate donne en même temps l'index i (0, 1, 2...) et le texte. On a besoin de i uniquement pour afficher la progression.
        print(f"   embedding {i+1}/{len(texts)}...", end="\r") # Affiche la progression dans le terminal.
        
        # L'appel à Ollama
        response = httpx.post( # envoie le texte à Ollama
            f"{settings.OLLAMA_BASE_URL}/api/embeddings",
            json={
                "model": EMBEDDING_MODEL,
                "prompt": text
            },
            timeout=30 # si Ollama ne répond pas en 30 secondes pour un embedding, on abandonne. Les embeddings sont rapides donc 30 secondes est largement suffisant.
        )
        # La réponse d'Ollama
        response.raise_for_status() # vérifie qu'il n'y a pas d'erreur
        embeddings.append(response.json()["embedding"]) # ajoute le vecteur à la boîte
        """ 
        Ollama retourne un JSON qui ressemble à :
        {
         "embedding": [0.21, -0.54, 0.87, 0.03, 0.69, -0.12, ...]
        }
        """

    print(f"   ✅ {len(embeddings)} embeddings générés")
    return embeddings # retourne tous les vecteurs