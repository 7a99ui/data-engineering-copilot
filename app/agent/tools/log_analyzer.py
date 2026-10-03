# re — expressions régulières pour parser les logs
import re

# datetime pour travailler avec les timestamps des logs
from datetime import datetime

# Path pour lire les fichiers de logs depuis le disque
from pathlib import Path


# Pattern regex pour parser une ligne de log standard
# Format attendu : [2026-10-01 02:41:23] ERROR message ici
#
# Explication du pattern :
# \[              → crochet ouvrant littéral (échappé car [ a un sens spécial en regex)
# (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})  → groupe 1 : timestamp "2026-10-01 02:41:23"
# \]              → crochet fermant littéral
# \s+             → un ou plusieurs espaces
# (INFO|WARNING|ERROR|CRITICAL|DEBUG)  → groupe 2 : niveau de log
# \s+             → espaces
# (.+)            → groupe 3 : le message (tout le reste de la ligne)
LOG_PATTERN = re.compile(
    r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s+(INFO|WARNING|ERROR|CRITICAL|DEBUG)\s+(.+)'
)


def parse_log_line(line: str) -> dict | None:
    """
    Parse une ligne de log et retourne un dict structuré.
    Retourne None si la ligne ne correspond pas au format attendu.

    Exemple :
    "[2026-10-01 02:41:45] ERROR OutOfMemoryError: Java heap space"
    →
    {
        "timestamp": "2026-10-01 02:41:45",
        "level"    : "ERROR",
        "message"  : "OutOfMemoryError: Java heap space"
    }
    """

    # re.match() essaie de faire correspondre le pattern au début de la ligne
    match = LOG_PATTERN.match(line.strip())

    if not match:
        # La ligne ne correspond pas au pattern → on l'ignore
        # ex: lignes vides, headers, stack traces...
        return None

    # Extraire les 3 groupes capturés par le pattern
    # group(1) → timestamp, group(2) → niveau, group(3) → message
    return {
        "timestamp": match.group(1),
        "level"    : match.group(2),
        "message"  : match.group(3).strip()
    }


def analyze_logs(log_content: str) -> dict:
    """
    Analyse un contenu de logs complet et extrait les informations clés.
    Filtre les erreurs, compte les niveaux, détecte les patterns récurrents.

    log_content : le contenu brut des logs (string multi-lignes)
    -> dict     : résumé structuré des logs pour le LLM
    """

    # Découper le contenu en lignes individuelles
    lines = log_content.strip().split("\n")

    # Listes pour accumuler les entrées par niveau
    all_entries  = []
    errors       = []
    warnings     = []
    criticals    = []

    # Parser chaque ligne
    for line in lines:
        parsed = parse_log_line(line)

        # Ignorer les lignes qui ne correspondent pas au format
        if parsed is None:
            continue

        all_entries.append(parsed)

        # Trier par niveau de sévérité
        if parsed["level"] == "ERROR":
            errors.append(parsed)
        elif parsed["level"] == "WARNING":
            warnings.append(parsed)
        elif parsed["level"] == "CRITICAL":
            criticals.append(parsed)

    # ── Détecter les patterns d'erreurs récurrents ──
    # On compte combien de fois chaque message d'erreur apparaît
    # pour identifier les erreurs récurrentes vs ponctuelles
    error_patterns = {}
    for entry in errors + criticals:
        msg = entry["message"]

        # Normaliser le message — enlever les détails variables
        # ex: "OutOfMemoryError at line 42" et "OutOfMemoryError at line 87"
        # doivent être regroupés comme le même pattern
        # On garde seulement les 50 premiers caractères comme clé
        key = msg[:50]
        error_patterns[key] = error_patterns.get(key, 0) + 1

    # Trier les patterns par fréquence décroissante
    sorted_patterns = sorted(
        error_patterns.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # ── Construire le résumé ──
    return {
        # Statistiques générales
        "total_lines"  : len(lines),
        "parsed_lines" : len(all_entries),
        "error_count"  : len(errors),
        "warning_count": len(warnings),
        "critical_count": len(criticals),

        # Les 10 erreurs les plus récentes
        # le LLM se concentre sur les dernières erreurs
        "recent_errors": errors[-10:] if errors else [],

        # Les erreurs critiques — priorité maximale
        "criticals"    : criticals,

        # Les patterns récurrents (max 5)
        # utile pour identifier les problèmes systémiques
        "top_error_patterns": sorted_patterns[:5],

        # Résumé textuel compact pour le LLM
        # au lieu de lui envoyer 1000 lignes de logs
        "summary": (
            f"Analyzed {len(lines)} log lines. "
            f"Found {len(errors)} errors, {len(warnings)} warnings, "
            f"{len(criticals)} critical issues. "
            f"Most frequent error: {sorted_patterns[0][0] if sorted_patterns else 'none'}."
        )
    }


def analyze_log_file(file_path: str) -> dict:
    """
    Lit un fichier de logs depuis le disque et l'analyse.
    Wrapper autour de analyze_logs() pour lire depuis un fichier.

    file_path : chemin vers le fichier de logs
    -> dict   : résumé structuré des logs
    """

    path = Path(file_path)

    # Vérifier que le fichier existe
    if not path.exists():
        return {
            "error"  : f"Log file not found: {file_path}",
            "summary": "No log file found at the specified path."
        }

    # Lire le contenu du fichier
    # encoding="utf-8" + errors="ignore" → ignore les caractères non-UTF8
    # utile pour les logs qui contiennent des bytes corrompus
    content = path.read_text(encoding="utf-8", errors="ignore")

    # Analyser le contenu
    return analyze_logs(content)


def analyze_log_string(log_content: str) -> dict:
    """
    Analyse un contenu de logs passé directement comme string.
    Utilisé quand les logs viennent de la DB ou d'une API
    plutôt que d'un fichier.

    log_content : contenu des logs en string
    -> dict     : résumé structuré
    """
    return analyze_logs(log_content)