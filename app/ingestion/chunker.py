def chunk_text(text: str, chunk_size: int = 1000,
               chunk_overlap: int = 100) -> list[str]:
    """
    Découpe un texte en morceaux de taille chunk_size
    avec un chevauchement de chunk_overlap caractères.
    """
    
    # chunks est la liste vide qu'on va remplir. start est le curseur — il indique où on en est dans le texte. Il commence à 0 (début du texte).
    chunks = []
    start = 0

    # On continue tant que le curseur n'a pas dépassé la fin du texte
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # strip() enlève les espaces et sauts de ligne au début et à la fin. Si après ça le chunk est vide, on ne l'ajoute pas. Sinon on l'ajoute à la liste.
        if chunk.strip():           # ignore les chunks vides
            chunks.append(chunk)

        # C'est la ligne la plus importante. Au lieu d'avancer de 500, on avance de 500 - 50 = 450. Ça veut dire que les 50 derniers caractères du chunk actuel seront répétés au début du prochain chunk — c'est l'overlap.
        start = end - chunk_overlap # recule de chunk_overlap
                                    # pour créer le chevauchement

    return chunks

# Cette fonction reçoit ce que le parser a produit et ajoute les métadonnées à chaque chunk.
# Elle prend un dictionnaire (la sortie du parser) et retourne une liste de dictionnaires enrichis.
def chunk_document(parsed_doc: dict) -> list[dict]:
    """
    Prend un document parsé et retourne une liste de chunks
    avec leurs métadonnées complètes.
    """
    
    # all_chunks accumule tous les chunks de toutes les pages. metadata récupère les infos du fichier — nom, type, chemin. Souviens-toi, le parser retourne {"content": [...], "metadata": {...}}.
    all_chunks = []
    metadata = parsed_doc["metadata"]

    # On boucle sur chaque page du document. Pour chaque page, on extrait le texte et le numéro de page, puis on appelle chunk_text() qu'on vient de décortiquer — elle retourne la liste des chunks de cette page.
    for page_data in parsed_doc["content"]:
        text = page_data["text"]
        page_num = page_data["page"]
        chunks = chunk_text(text)

        # enumerate donne en même temps l'index i (0, 1, 2...) et le contenu chunk. C'est utile pour savoir que c'est le "3ème chunk de la page 2".
        for i, chunk in enumerate(chunks):
            # Pour chaque chunk, on crée un dictionnaire avec le texte ET toutes les métadonnées
            all_chunks.append({
                "text": chunk,
                "metadata": {
                    **metadata,             # source, file_type...
                    "page": page_num,
                    "chunk_index": i
                }
            })

    return all_chunks