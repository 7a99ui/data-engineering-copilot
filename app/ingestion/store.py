# httpx pour faire des requêtes HTTP vers Qdrant (comme pour Ollama)
import httpx

# uuid génère des identifiants uniques aléatoires
# ex: "f47ac10b-58cc-4372-a567-0e02b2c3d479"
import uuid

# settings pour récupérer QDRANT_HOST et QDRANT_PORT depuis le .env
from app.config import settings

# Le nom de la "table" dans Qdrant — comme un nom de table en SQL
COLLECTION_NAME = "documents"

# La taille des vecteurs produits par nomic-embed-text
# DOIT correspondre exactement au modèle dans embedder.py
# si tu changes de modèle → tu changes ce nombre
VECTOR_SIZE = 768

# L'URL de base de Qdrant construite depuis le .env
# résultat : "http://localhost:6333"
# tous les appels HTTP commenceront par cette URL
BASE_URL = f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}"


def create_collection_if_not_exists():
    """
    Crée la collection Qdrant si elle n'existe pas encore.
    Appelée UNE SEULE FOIS au début de l'ingestion.
    Analogue à CREATE TABLE IF NOT EXISTS en SQL.
    """

    # On demande à Qdrant : "est-ce que cette collection existe ?"
    # GET /collections/documents → 200 si elle existe, 404 si elle n'existe pas
    response = httpx.get(f"{BASE_URL}/collections/{COLLECTION_NAME}")

    # Si Qdrant répond 200 → la collection existe déjà
    # on sort de la fonction sans rien faire (return arrête la fonction ici)
    if response.status_code == 200:
        print(f"ℹ️  Collection '{COLLECTION_NAME}' déjà existante")
        return

    # Si on arrive ici → la collection n'existe pas (code 404)
    # On la crée avec PUT /collections/documents
    # PUT = "crée ou remplace" en HTTP
    response = httpx.put(
        f"{BASE_URL}/collections/{COLLECTION_NAME}",

        # Le corps JSON dit à Qdrant comment configurer la collection
        json={
            "vectors": {
                # Chaque vecteur aura exactement 768 nombres
                # Qdrant refusera tout vecteur d'une autre taille
                "size": VECTOR_SIZE,

                # La méthode de comparaison entre vecteurs
                # "Cosine" = similarité cosine (angle entre vecteurs)
                # c'est la meilleure méthode pour comparer du texte
                "distance": "Cosine"
            }
        }
    )

    # Si Qdrant répond avec une erreur (400, 500...)
    # raise_for_status() lève une exception Python claire
    # sans ça, le code continuerait même en cas d'échec silencieux
    response.raise_for_status()
    print(f"✅ Collection '{COLLECTION_NAME}' créée")


def store_chunks(chunks: list[dict], embeddings: list[list[float]]):
    """
    Insère les chunks et leurs vecteurs dans Qdrant.
    chunks     = liste de dicts {text, metadata} produits par chunker.py
    embeddings = liste de vecteurs produits par embedder.py
    Les deux listes ont la même taille — chunk[i] correspond à embedding[i]
    """

    # Liste vide qui va accumuler tous les points avant insertion
    # On attend d'avoir TOUS les points pour les envoyer en une seule requête
    # c'est plus efficace que d'envoyer point par point
    points = []

    # zip() associe chaque chunk à son vecteur correspondant
    # chunk[0] ↔ embedding[0], chunk[1] ↔ embedding[1], etc.
    # comme fermer une fermeture éclair entre les deux listes
    for chunk, embedding in zip(chunks, embeddings):

        # On construit un "point" — le format qu'attend Qdrant
        # Un point = ID unique + vecteur + données associées
        points.append({

            # ID unique pour ce point dans Qdrant
            # uuid4() génère un identifiant aléatoire garanti unique
            # str() le convertit en chaîne car Qdrant attend une string
            # ex: "f47ac10b-58cc-4372-a567-0e02b2c3d479"
            "id": str(uuid.uuid4()),

            # Le vecteur de 768 nombres qui représente le sens du texte
            # produit par embedder.py via Ollama/nomic-embed-text
            # c'est ce que Qdrant utilisera pour comparer avec les questions
            "vector": embedding,

            # Le payload = tout ce qui accompagne le vecteur
            # Qdrant le stocke et le retourne quand il trouve ce point
            # au Sprint 3, c'est ce qu'on donnera au LLM comme contexte
            "payload": {

                # Le texte original du chunk — la vraie valeur
                # c'est ce texte qu'on enverra au LLM pour répondre
                "text": chunk["text"],

                # **chunk["metadata"] = spread operator
                # "déverse" tous les champs de metadata ici
                # équivalent à écrire :
                #   "source": chunk["metadata"]["source"],
                #   "file_type": chunk["metadata"]["file_type"],
                #   "page": chunk["metadata"]["page"],
                #   "chunk_index": chunk["metadata"]["chunk_index"]
                **chunk["metadata"]
            }
        })

    # On envoie TOUS les points en une seule requête HTTP
    # PUT /collections/documents/points = insérer des points dans la collection
    # json={"points": points} = le corps de la requête avec tous nos points
    # timeout=60 = si Qdrant ne répond pas en 60 secondes → erreur
    # (60s car on peut envoyer beaucoup de points d'un coup)
    response = httpx.put(
        f"{BASE_URL}/collections/{COLLECTION_NAME}/points",
        json={"points": points},
        timeout=60
    )

    # Vérifie que Qdrant a bien accepté tous les points
    # si une erreur s'est produite → lève une exception claire
    response.raise_for_status()
    print(f"✅ {len(points)} chunks insérés dans Qdrant")