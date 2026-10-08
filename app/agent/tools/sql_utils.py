# Expressions régulières pour extraire et nettoyer les requêtes SQL
import re


def extract_sql(llm_output: str) -> str:
    """
    Extrait la requête SQL depuis la sortie brute du LLM.
    Gère tous les cas courants :
    - Texte avant la requête ("SQL Query (SELECT only): SELECT...")
    - Bloc markdown (```sql SELECT...```)
    - Texte après la requête ("This query counts...")
    - Requête WITH ... SELECT (CTEs)
    - Double instruction (SELECT 1; DROP TABLE → coupe au premier ;)

    llm_output : la sortie brute du LLM
    -> str     : la requête SQL propre, sans texte parasite ni point-virgule
    """

    text = llm_output.strip()

    # ── Étape 1 : extraire le contenu d'un bloc markdown ```sql ... ``` ──
    # Le LLM met parfois la requête dans des backticks malgré la consigne
    # re.S = DOTALL → le point matche aussi les sauts de ligne
    block = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
    if block:
        # On a trouvé un bloc markdown → travailler uniquement son contenu
        text = block.group(1).strip()

    # ── Étape 2 : chercher la première LIGNE qui commence par SELECT ou WITH ──
    # re.M = MULTILINE → ^ matche le début de chaque ligne (pas seulement du texte)
    # On cherche d'abord en début de ligne (cas normal)
    start = re.search(r"^\s*(SELECT|WITH)\b", text, re.I | re.M)

    if not start:
        # Fallback : chercher SELECT ou WITH n'importe où dans le texte
        # (cas où le LLM met tout sur une ligne avec du texte avant)
        start = re.search(r"\b(SELECT|WITH)\b", text, re.I)

    if not start:
        # Aucun SELECT ni WITH trouvé → retourner vide
        # validate_query() refusera proprement avec un message clair
        return ""

    # Couper tout ce qui précède le SELECT/WITH
    text = text[start.start():]

    # ── Étape 3 : couper au premier point-virgule ──
    # Deux raisons :
    # 1. Évite les doubles instructions (SELECT 1; DROP TABLE...)
    # 2. Notre execute_query enveloppe dans (...) AS subq — un ; intermédiaire casserait tout
    # split(";")[0] prend tout avant le premier ;
    sql = text.split(";")[0].strip()

    return sql