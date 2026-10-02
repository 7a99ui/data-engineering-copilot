# TypedDict — permet de définir un dictionnaire avec des types précis
# LangGraph utilise TypedDict pour définir la structure de l'état
from typing import TypedDict, Optional


class AgentState(TypedDict):
    """
    L'état partagé entre tous les nœuds du graphe.
    Chaque nœud peut lire et écrire dans cet état.
    C'est le "tableau blanc" qui circule à travers le graphe.

    TypedDict = dictionnaire Python avec types définis
    → LangGraph sait exactement quels champs existent
    → ton éditeur de code peut faire de l'autocomplétion
    → les erreurs de type sont détectées tôt
    """

    # La question posée par l'utilisateur
    # Définie au début, ne change jamais pendant l'exécution
    question: str

    # L'intent classifié par analyze_intent
    # Valeurs possibles : "rag_search", "direct_answer", "chitchat"
    # Optional car pas encore défini au début du graphe
    intent: Optional[str]

    # Les chunks récupérés par rag_search
    # Liste de dicts avec "score", "payload"
    # None si pas de recherche RAG effectuée
    chunks: Optional[list]

    # La réponse finale générée
    # Remplie par rag_search, direct_answer, ou chitchat
    answer: Optional[str]

    # Les sources utilisées pour répondre
    # Liste de dicts avec source, page, score, chunk
    # None si pas de RAG (direct_answer ou chitchat)
    sources: Optional[list]

    # Les étapes parcourues dans le graphe
    # Utile pour le debug et la traçabilité
    # ex: ["analyze_intent", "rag_search", "format_response"]
    steps: list