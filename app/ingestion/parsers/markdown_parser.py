from pathlib import Path

def parse_markdown(file_path: str) -> dict:
    """
    Lit un fichier Markdown et retourne le texte
    avec ses métadonnées.
    """
    path = Path(file_path)

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    return {
        "content": [{"text": text, "page": 1}],
        "metadata": {
            "source": path.name,
            "file_type": "markdown",
            "total_pages": 1,
            "file_path": str(path)
        }
    }