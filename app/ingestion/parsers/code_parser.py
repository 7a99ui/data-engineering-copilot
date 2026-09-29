from pathlib import Path

SUPPORTED_EXTENSIONS = {
    ".py": "python",
    ".sql": "sql",
    ".js": "javascript",
    ".ts": "typescript",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json"
}

def parse_code(file_path: str) -> dict:
    """
    Lit un fichier de code et retourne le texte
    avec ses métadonnées, y compris le langage.
    """
    path = Path(file_path)
    language = SUPPORTED_EXTENSIONS.get(path.suffix, "unknown")

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    return {
        "content": [{"text": text, "page": 1}],
        "metadata": {
            "source": path.name,
            "file_type": "code",
            "language": language,
            "total_pages": 1,
            "file_path": str(path)
        }
    }