import pypdf
from pathlib import Path

def parse_pdf(file_path: str) -> dict:
    """
    Lit un fichier PDF et retourne le texte extrait
    avec ses métadonnées.
    """
    path = Path(file_path)
    text_pages = []

    with open(path, "rb") as f:
        reader = pypdf.PdfReader(f)

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()

            if text and text.strip():   # ignore les pages vides
                text_pages.append({
                    "text": text.strip(),
                    "page": page_num + 1
                })

    return {
        "content": text_pages,
        "metadata": {
            "source": path.name,
            "file_type": "pdf",
            "total_pages": len(reader.pages),
            "file_path": str(path)
        }
    }