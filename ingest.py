import argparse
from pathlib import Path
from app.ingestion.parsers.pdf_parser import parse_pdf
from app.ingestion.parsers.markdown_parser import parse_markdown
from app.ingestion.parsers.code_parser import parse_code
from app.ingestion.chunker import chunk_document
from app.ingestion.embedder import embed_texts
from app.ingestion.store import create_collection_if_not_exists, store_chunks

# association extension → parser
PARSERS = {
    ".pdf": parse_pdf,
    ".md":  parse_markdown,
    ".py":  parse_code,
    ".sql": parse_code,
}

def ingest(source_dir: str):
    print(f"🔍 Recherche dans : {source_dir}")   # ← ajoute
    source = Path(source_dir)
    files = [f for f in source.rglob("*") if f.suffix in PARSERS]
    print(f"📂 Fichiers trouvés : {files}")

    if not files:
        print("❌ Aucun fichier supporté trouvé")
        return

    print(f"\n📂 {len(files)} fichier(s) trouvé(s)\n")
    create_collection_if_not_exists()

    all_chunks = []

    # 1. Parser + Chunker
    for file in files:
        print(f"🔍 Parsing : {file.name}")
        parser = PARSERS[file.suffix]
        parsed = parser(str(file))
        chunks = chunk_document(parsed)
        all_chunks.extend(chunks)
        print(f"   → {len(chunks)} chunks créés")

    print(f"\n✂️  Total : {len(all_chunks)} chunks\n")

    # 2. Embeddings
    print("🔢 Génération des embeddings...")
    texts = [c["text"] for c in all_chunks]
    embeddings = embed_texts(texts)

    # 3. Stockage
    print("\n💾 Stockage dans Qdrant...")
    store_chunks(all_chunks, embeddings)

    print(f"\n🎉 Ingestion terminée — {len(all_chunks)} chunks prêts")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="./docs",
                        help="Dossier contenant les documents")
    args = parser.parse_args()
    ingest(args.source)