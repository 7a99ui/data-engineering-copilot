# psycopg2 — le driver Python officiel pour PostgreSQL
# comme httpx pour Ollama, psycopg2 fait le pont entre Python et PostgreSQL
import psycopg2

# psycopg2.extras donne des curseurs qui retournent des dicts au lieu de tuples
# plus facile à manipuler : row["table_name"] au lieu de row[0]
import psycopg2.extras

# settings pour récupérer les infos de connexion depuis le .env
from app.config import settings


def get_connection():
    """
    Crée et retourne une connexion à PostgreSQL.
    Appelée à chaque utilisation — pas de connexion persistante
    pour éviter les problèmes de timeout en production.

    Les paramètres viennent tous du fichier .env via settings :
    host     = POSTGRES_HOST (localhost)
    port     = POSTGRES_PORT (5432)
    database = POSTGRES_DB
    user     = POSTGRES_USER
    password = POSTGRES_PASSWORD
    """
    return psycopg2.connect(
        host    =settings.POSTGRES_HOST,
        port    =settings.POSTGRES_PORT,
        database=settings.POSTGRES_DB,
        user    =settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD
    )


def get_all_tables() -> list[dict]:
    """
    Retourne la liste de toutes les tables dans le schéma public.

    information_schema.tables est une table spéciale que PostgreSQL
    maintient automatiquement — elle décrit la structure de la DB.
    C'est comme les métadonnées de la base de données.

    Retourne une liste de dicts :
    [{"table_name": "customers", "table_type": "BASE TABLE"}, ...]
    """

    # "with" garantit que la connexion est fermée même si une erreur survient
    # c'est le même pattern que "async with httpx.AsyncClient()"
    with get_connection() as conn:

        # cursor_factory=RealDictCursor → chaque ligne retournée est un dict
        # sans ça : row = ("customers", "BASE TABLE")  → row[0] peu lisible
        # avec ça  : row = {"table_name": "customers", "table_type": "BASE TABLE"}
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:

            # information_schema.tables = table PostgreSQL qui liste toutes les tables
            # WHERE table_schema = 'public' → ignorer les tables système internes
            # ORDER BY table_name → résultats triés alphabétiquement
            cur.execute("""
                SELECT
                    table_name,
                    table_type
                FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name
            """)

            # fetchall() récupère TOUTES les lignes d'un coup
            # chaque ligne est un dict grâce à RealDictCursor
            rows = cur.fetchall()

            # Convertir chaque RealDictRow en dict Python standard
            # pour éviter des problèmes de sérialisation JSON plus tard
            return [dict(row) for row in rows]


def get_table_schema(table_name: str) -> list[dict]:
    """
    Retourne le schéma d'une table spécifique :
    colonnes, types de données, et si elles peuvent être NULL.

    information_schema.columns contient les métadonnées de chaque colonne
    de chaque table dans la base de données.

    table_name : le nom de la table à inspecter
    -> list[dict] : liste des colonnes avec leurs propriétés
    """

    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:

            # %s = paramètre positif — psycopg2 l'échappe automatiquement
            # CRUCIAL : ne jamais construire une requête SQL avec f-string
            # car ça ouvre la porte aux injections SQL
            # ❌ f"WHERE table_name = '{table_name}'"  → injection SQL possible
            # ✅ "WHERE table_name = %s", (table_name,) → échappé automatiquement
            cur.execute("""
                SELECT
                    column_name,
                    data_type,
                    is_nullable,
                    column_default
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = %s
                ORDER BY ordinal_position
            """, (table_name,))   # tuple avec la valeur — psycopg2 gère l'échappement

            rows = cur.fetchall()
            return [dict(row) for row in rows]


def get_table_sample(table_name: str, limit: int = 3) -> list[dict]:
    """
    Retourne quelques exemples de lignes d'une table.
    Utile pour que le LLM comprenne le contenu de la table.

    ATTENTION : on construit ici le nom de la table dans la requête
    car les paramètres psycopg2 ne fonctionnent pas pour les noms de tables.
    On valide donc le nom de la table manuellement pour éviter les injections.

    table_name : nom de la table
    limit      : nombre de lignes à retourner (défaut 3)
    """

    # Validation du nom de table — sécurité
    # on vérifie que la table existe vraiment dans la DB
    # avant de la mettre dans une requête SQL
    existing_tables = [t["table_name"] for t in get_all_tables()]
    if table_name not in existing_tables:
        # Si la table n'existe pas → retourner une liste vide
        # plutôt que de lever une exception qui casserait l'agent
        return []

    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:

            # Ici on peut utiliser f-string car on a validé table_name au-dessus
            # le nom de table a été vérifié contre la vraie liste des tables
            cur.execute(f"SELECT * FROM {table_name} LIMIT %s", (limit,))

            rows = cur.fetchall()
            # Convertir en dict Python — chaque valeur est sérialisable JSON
            return [dict(row) for row in rows]


def inspect_database() -> dict:
    """
    Fonction principale du DB Inspector.
    Retourne une vue complète de la base de données :
    liste des tables + schéma de chaque table + exemples.

    C'est cette fonction que le nœud LangGraph appellera.
    Elle agrège tout ce dont le LLM a besoin pour répondre
    aux questions sur la structure de la DB.
    """

    # Récupérer toutes les tables
    tables = get_all_tables()

    # Pour chaque table, récupérer son schéma et quelques exemples
    tables_info = []
    for table in tables:
        name = table["table_name"]

        # Schéma complet de la table (colonnes, types...)
        schema = get_table_schema(name)

        # 3 exemples de lignes pour que le LLM comprenne les données
        sample = get_table_sample(name, limit=3)

        tables_info.append({
            "table_name" : name,
            "table_type" : table["table_type"],
            "columns"    : schema,
            "sample_rows": sample,
            "row_count"  : len(sample)   # approximatif — juste pour info
        })

    return {
        "database"    : settings.POSTGRES_DB,
        "tables_count": len(tables),
        "tables"      : tables_info
    }