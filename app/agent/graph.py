# LangGraph — les imports principaux
from langgraph.graph import StateGraph, END

# Notre état partagé
from app.agent.state import AgentState

# Nos nœuds
from app.agent.nodes import (
    analyze_intent_node,
    rag_search_node,
    direct_answer_node,
    chitchat_node,
    format_response_node
)

# Notre router conditionnel
from app.agent.router import route_after_intent


def create_agent_graph():
    """
    Construit et compile le graphe LangGraph.
    Retourne un graphe compilé prêt à être invoqué.

    Le graphe définit :
    - les nœuds (quoi faire)
    - les edges (dans quel ordre)
    - les edges conditionnels (selon quoi décider)
    """

    # ── Créer le graphe avec notre état ──
    # StateGraph prend le type de l'état en paramètre
    # il sait ainsi quels champs existent et lesquels mettre à jour
    builder = StateGraph(AgentState)

    # ── Ajouter les nœuds ──
    # add_node(nom, fonction)
    # nom = identifiant du nœud dans le graphe
    # fonction = ce qui s'exécute quand ce nœud est activé
    builder.add_node("analyze_intent",  analyze_intent_node)
    builder.add_node("rag_search",      rag_search_node)
    builder.add_node("direct_answer",   direct_answer_node)
    builder.add_node("chitchat",        chitchat_node)
    builder.add_node("format_response", format_response_node)

    # ── Définir le point d'entrée ──
    # Le premier nœud à exécuter quand on invoke le graphe
    builder.set_entry_point("analyze_intent")

    # ── Edge conditionnel après analyze_intent ──
    # add_conditional_edges(nœud_source, fonction_routing, mapping)
    # après "analyze_intent", appelle route_after_intent(state)
    # selon la valeur retournée, va au nœud correspondant
    builder.add_conditional_edges(
        # Nœud source — après ce nœud, on décide
        "analyze_intent",

        # Fonction qui retourne le nom du prochain nœud
        # reçoit l'état et retourne "rag_search", "direct_answer", ou "chitchat"
        route_after_intent,

        # Mapping explicite des valeurs possibles → nœuds
        # obligatoire pour que LangGraph sache quels nœuds sont possibles
        {
            "rag_search"    : "rag_search",
            "direct_answer" : "direct_answer",
            "chitchat"      : "chitchat"
        }
    )

    # ── Edges fixes vers format_response ──
    # Tous les chemins convergent vers format_response
    # add_edge(source, destination) = toujours aller de source à destination
    builder.add_edge("rag_search",    "format_response")
    builder.add_edge("direct_answer", "format_response")
    builder.add_edge("chitchat",      "format_response")

    # ── Edge final ──
    # Après format_response → fin du graphe
    builder.add_edge("format_response", END)

    # ── Compiler le graphe ──
    # compile() vérifie la cohérence du graphe
    # (pas de nœuds orphelins, pas de boucles infinies...)
    # et retourne un objet invocable
    return builder.compile()


# Instance unique du graphe — créée au démarrage de l'app
# pattern Singleton — le graphe n'est compilé qu'une fois
agent_graph = create_agent_graph()