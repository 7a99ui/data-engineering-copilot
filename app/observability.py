# uuid — génère l'identifiant de chaque trace
import uuid

# LangFuse SDK — client d'observabilité
# Gère la connexion, le buffering et l'envoi asynchrone des traces
from langfuse import Langfuse

# settings pour récupérer les clés depuis le .env
from app.config import settings


def is_enabled() -> bool:
    """Retourne True si LangFuse est configuré et actif."""
    return bool(settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY)


# ── Initialisation du client LangFuse ──
# Ce client est partagé par tous les nœuds — pattern Singleton
# Il bufferise les traces et les envoie en arrière-plan
# Si les clés sont vides → LangFuse est désactivé silencieusement
langfuse = Langfuse(
    public_key=settings.LANGFUSE_PUBLIC_KEY,
    secret_key=settings.LANGFUSE_SECRET_KEY,
    host      =settings.LANGFUSE_HOST,
    enabled   =is_enabled()
)


def new_trace(**kwargs):
    """
    Crée une nouvelle trace.
    Retourne (trace_id, trace) — ou (None, None) si LangFuse est désactivé.

    Seul trace_id (un texte) doit être stocké dans l'état LangGraph :
    l'objet trace n'est pas sérialisable par le checkpointer.
    """
    if not is_enabled():
        return None, None
    trace_id = str(uuid.uuid4())
    return trace_id, langfuse.trace(id=trace_id, **kwargs)


def get_trace(state: dict):
    """
    Reconstruit l'objet trace à partir de l'ID stocké dans l'état.
    Retourne None si LangFuse est désactivé ou si l'état n'a pas d'ID.
    Utiliser le même id = continuer la même trace.
    """
    trace_id = state.get("_lf_trace")
    if is_enabled() and trace_id:
        return langfuse.trace(id=trace_id)
    return None