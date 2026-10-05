# LangGraph — les imports principaux
from langgraph.graph import StateGraph, END

# MemorySaver — sauvegarde l'état entre deux appels pour le Human in the Loop
# stocke en RAM — suffisant pour les tests
# en production on utiliserait SqliteSaver ou PostgresSaver pour la persistance
from langgraph.checkpoint.memory import MemorySaver

# Notre état partagé — TypedDict avec tous les champs Sprint 5 + Sprint 6
from app.agent.state import AgentState

# Tous les nœuds du graphe
from app.agent.nodes import (
    # ── Nœuds Sprint 5 ──
    analyze_intent_node,   # classifie l'intent de la question
    rag_search_node,       # RAG hybride Sprint 4
    direct_answer_node,    # LLM seul pour questions générales
    chitchat_node,         # réponse conversationnelle courte
    format_response_node,  # dernier nœud — tous y convergent

    # ── Nœuds Sprint 6 ──
    db_inspect_node,       # inspecte la structure PostgreSQL
    sql_query_node,        # génère la requête SQL + attend confirmation
    sql_execute_node,      # exécute la requête après confirmation humaine
    log_analyzer_node,     # analyse les logs de pipeline
)

# Fonctions de routing conditionnel
from app.agent.router import route_after_intent, route_after_sql


def create_agent_graph():
    """
    Construit et compile le graphe LangGraph complet.
    Sprint 6 : 9 nœuds, 5 intents, Human in the Loop pour SQL.

    Le graphe définit :
    - les nœuds (quoi faire à chaque étape)
    - les edges fixes (transitions certaines)
    - les edges conditionnels (transitions selon l'état)
    - l'interruption avant sql_execute (Human in the Loop)
    """

    # ── Créer le graphe avec notre état ──
    # StateGraph prend le TypedDict en paramètre
    # LangGraph sait ainsi quels champs existent dans l'état
    # et peut valider les mises à jour de chaque nœud
    builder = StateGraph(AgentState)

    # ────────────────────────────────────
    # ENREGISTREMENT DES NŒUDS
    # add_node(nom_string, fonction_async)
    # nom_string = identifiant utilisé dans les edges
    # fonction   = ce qui s'exécute quand ce nœud est activé
    # ────────────────────────────────────

    # Nœud commun — toujours le premier à s'exécuter
    builder.add_node("analyze_intent",  analyze_intent_node)

    # Nœuds Sprint 5 — 3 chemins selon l'intent
    builder.add_node("rag_search",      rag_search_node)
    builder.add_node("direct_answer",   direct_answer_node)
    builder.add_node("chitchat",        chitchat_node)

    # Nœuds Sprint 6 — 3 nouveaux outils
    builder.add_node("db_inspect",      db_inspect_node)
    builder.add_node("sql_query_node",       sql_query_node)     # génère SQL + attend
    builder.add_node("sql_execute_node",     sql_execute_node)   # exécute après confirmation
    builder.add_node("log_analyzer",    log_analyzer_node)

    # Nœud final — tous les chemins y convergent
    builder.add_node("format_response", format_response_node)

    # ────────────────────────────────────
    # POINT D'ENTRÉE
    # Le premier nœud exécuté à chaque invoke()
    # ────────────────────────────────────
    builder.set_entry_point("analyze_intent")

    # ────────────────────────────────────
    # EDGE CONDITIONNEL PRINCIPAL
    # Après analyze_intent → route_after_intent(state)
    # décide vers quel nœud aller selon l'intent classifié
    # ────────────────────────────────────
    builder.add_conditional_edges(
        # Nœud source — après ce nœud, on appelle la fonction de routing
        "analyze_intent",

        # Fonction de routing — reçoit l'état, retourne un nom de nœud
        # définie dans router.py — gère 5 intents au Sprint 6
        route_after_intent,

        # Mapping explicite des valeurs possibles → nœuds
        # OBLIGATOIRE pour que LangGraph sache quels nœuds sont atteignables
        # si route_after_intent retourne une valeur hors de ce mapping → erreur
        {
            "rag_search"   : "rag_search",    # question technique → RAG hybride
            "direct_answer": "direct_answer", # question générale → LLM seul
            "chitchat"     : "chitchat",      # salutation → réponse courte
            "db_inspect"   : "db_inspect",    # ← Sprint 6 : structure DB
            "sql_query"    : "sql_query_node",     # ← Sprint 6 : exécution SQL
            "log_analyzer" : "log_analyzer",  # ← Sprint 6 : analyse logs
        }
    )

    # ────────────────────────────────────
    # EDGE CONDITIONNEL SQL
    # Après sql_query_node → route_after_sql(state)
    # décide si on exécute la requête ou si on attend la confirmation
    # ────────────────────────────────────
    builder.add_conditional_edges(
        # Nœud source
        "sql_query_node",

        # Fonction de routing SQL — vérifie sql_approved dans l'état
        # sql_approved = None  → "sql_execute" (sera interrompu par interrupt_before)
        # sql_approved = True  → "sql_execute" (reprend après confirmation)
        # sql_approved = False → "format_response" (requête refusée)
        route_after_sql,

        {
            "sql_execute_node"    : "sql_execute_node",
            "format_response": "format_response"
        }
    )

    # ────────────────────────────────────
    # EDGES FIXES
    # Transitions certaines — pas de condition
    # tous les chemins convergent vers format_response
    # ────────────────────────────────────

    # Nœuds Sprint 5 → format_response
    builder.add_edge("rag_search",    "format_response")
    builder.add_edge("direct_answer", "format_response")
    builder.add_edge("chitchat",      "format_response")

    # Nœuds Sprint 6 → format_response
    builder.add_edge("db_inspect",    "format_response")
    builder.add_edge("sql_execute_node",   "format_response")
    builder.add_edge("log_analyzer",  "format_response")

    # format_response → END
    # Dernier nœud du graphe — après lui, le graphe se termine
    builder.add_edge("format_response", END)

    # ────────────────────────────────────
    # CHECKPOINTER — mémoire pour Human in the Loop
    # MemorySaver stocke l'état en RAM entre deux appels
    # Identifié par thread_id dans la config
    # → chaque conversation a sa propre mémoire
    # ────────────────────────────────────
    memory = MemorySaver()

    # ────────────────────────────────────
    # COMPILATION
    # compile() vérifie la cohérence du graphe :
    # - pas de nœuds enregistrés mais jamais atteints
    # - pas de destinations dans les edges qui n'existent pas
    # - pas de boucles infinies non intentionnelles
    #
    # interrupt_before=["sql_execute"] →
    # le graphe se met en PAUSE juste avant sql_execute
    # l'état est sauvegardé dans memory (MemorySaver)
    # le graphe reprend quand on appelle ainvoke(None, config)
    # ────────────────────────────────────
    return builder.compile(
        checkpointer=memory,
        interrupt_before=["sql_execute_node"]  # ← Human in the Loop
    )


# ── Instance unique du graphe compilé ──
# Pattern Singleton — compilé une seule fois au démarrage de l'app
# partagé par toutes les requêtes → économise le temps de compilation
# importé dans main.py : from app.agent.graph import agent_graph
agent_graph = create_agent_graph()