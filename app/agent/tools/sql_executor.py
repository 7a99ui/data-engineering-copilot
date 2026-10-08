# re — expressions régulières pour valider et parser les requêtes SQL
import re

# psycopg2 pour exécuter les requêtes validées
import psycopg2
import psycopg2.extras

# settings pour la connexion PostgreSQL
from app.config import settings

# get_connection depuis db_inspector — réutilisation du code
from app.agent.tools.db_inspector import get_connection


# Liste des mots-clés SQL dangereux — toute requête contenant ces mots
# sera automatiquement refusée AVANT d'être envoyée à PostgreSQL
# Cette liste couvre les opérations destructives ou de modification de données
DANGEROUS_KEYWORDS = [
    "DROP",     # supprime une table, une DB, un index...
    "DELETE",   # supprime des lignes
    "TRUNCATE", # vide une table entièrement
    "UPDATE",   # modifie des lignes
    "INSERT",   # ajoute des lignes
    "ALTER",    # modifie la structure d'une table
    "CREATE",   # crée une table, une DB...
    "GRANT",    # donne des permissions
    "REVOKE",   # retire des permissions
    "EXEC",     # exécute du code stocké
    "EXECUTE",  # idem
]


def validate_query(query: str) -> tuple[bool, str]:
    """
    Valide qu'une requête SQL est safe à exécuter.
    Retourne (True, "") si safe, (False, "raison") si dangereuse.

    C'est la première ligne de défense de notre SQL Executor.
    Principe de moindre privilège : l'agent ne peut que lire.

    query : la requête SQL à valider
    -> tuple[bool, str] : (is_safe, error_message)
    """

    # Normaliser la requête pour la validation
    # upper() → tout en majuscules pour comparer avec DANGEROUS_KEYWORDS
    # strip() → enlever les espaces au début et à la fin
    query_upper = query.upper().strip()

    # ── Vérification 1 : la requête doit commencer par SELECT ──
    # Une requête safe commence TOUJOURS par SELECT
    # Si elle commence par autre chose → refus immédiat
    if not query_upper.startswith("SELECT"):
        return False, f"Only SELECT queries are allowed. Query starts with: {query_upper[:20]}"

    # ── Vérification 2 : aucun mot-clé dangereux ──
    # Même si la requête commence par SELECT, elle pourrait contenir
    # des sous-requêtes dangereuses comme :
    # SELECT * FROM t; DROP TABLE t; -- injection SQL
    for keyword in DANGEROUS_KEYWORDS:
        # \b = word boundary → évite de matcher "SELECTIONS" pour "SELECT"
        # re.search → cherche le pattern n'importe où dans la requête
        if re.search(r'\b' + keyword + r'\b', query_upper):
            return False, f"Dangerous keyword detected: {keyword}"

    # ── Vérification 3 : pas de commentaires SQL ──
    # Les commentaires -- ou /* */ peuvent être utilisés pour cacher du code
    # ex : SELECT * FROM t; --DROP TABLE t (le DROP est commenté mais suspect)
    if "--" in query or "/*" in query:
        return False, "SQL comments are not allowed for security reasons"

    # ── Vérification 4 : une seule requête (pas de point-virgule intermédiaire) ──
    # "SELECT 1; DROP TABLE t" contient deux requêtes — on n'en veut qu'une
    # On enlève le ; final qui est souvent ajouté par habitude
    clean_query = query.rstrip().rstrip(";")
    if ";" in clean_query:
        return False, "Multiple statements are not allowed"

    # Si toutes les vérifications passent → requête safe
    return True, ""


def execute_query(query: str, max_rows: int = 50) -> dict:
    """
    Exécute une requête SQL validée et retourne les résultats.

    Cette fonction est appelée APRÈS la confirmation humaine.
    La validation a déjà été faite par validate_query().

    query    : la requête SELECT validée et approuvée
    max_rows : limite de lignes retournées (défaut 50)
               évite de saturer le contexte du LLM avec trop de données

    -> dict : {success, rows, columns, row_count, error}
    """

    # Double vérification — on valide encore même si déjà fait
    # principe de défense en profondeur : vérifier à chaque couche
    is_safe, error_msg = validate_query(query)
    if not is_safe:
        return {
            "success"  : False,
            "rows"     : [],
            "columns"  : [],
            "row_count": 0,
            "error"    : error_msg
        }

    # Exécuter la requête dans un bloc try/except
    # pour capturer les erreurs SQL (table inexistante, syntaxe incorrecte...)
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:

                # Exécuter la requête validée
                # On ajoute LIMIT pour éviter de récupérer des millions de lignes
                # même si l'utilisateur a oublié de mettre LIMIT dans sa requête
                # Si la requête se termine par ";" → ça casse "SELECT * FROM (...;) AS subq"
                clean_query = query.rstrip().rstrip(";").strip()
                # Envelopper proprement sans point-virgule intermédiaire
                limited_query = f"SELECT * FROM ({clean_query}) AS subq LIMIT {max_rows}"
                cur.execute(limited_query)

                # Récupérer toutes les lignes
                rows = cur.fetchall()

                # Extraire les noms des colonnes depuis le curseur
                # cur.description contient les métadonnées des colonnes
                columns = [desc[0] for desc in cur.description] if cur.description else []

                return {
                    "success"  : True,
                    "rows"     : [dict(row) for row in rows],
                    "columns"  : columns,
                    "row_count": len(rows),
                    "error"    : None
                }

    except psycopg2.Error as e:
        # Capturer les erreurs PostgreSQL spécifiques
        # ex: table inexistante, colonne incorrecte, syntaxe SQL invalide
        return {
            "success"  : False,
            "rows"     : [],
            "columns"  : [],
            "row_count": 0,
            "error"    : str(e)
        }