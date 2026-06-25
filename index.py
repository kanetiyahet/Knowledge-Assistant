"""
index_books.py
--------------
Indexes all books in ./books folder into ChromaDB using real semantic embeddings.

Supports: .pdf, .epub, .txt, .md, .json, .docx, .html, .csv

Run once (or re-run to add new books):
    pip install chromadb ollama pymupdf python-docx ebooklib beautifulsoup4 tqdm
    ollama pull nomic-embed-text
    python index_books.py
"""

import os, re, json, hashlib, time
from pathlib import Path

import chromadb
import fitz                          # pymupdf  → PDF
import docx                          # python-docx → DOCX
from ebooklib import epub            # ebooklib → EPUB
from bs4 import BeautifulSoup        # beautifulsoup4
import ollama
from tqdm import tqdm


# ─── CONFIG ──────────────────────────────────────────────────────────────────
BOOKS_FOLDER   = Path("books")
DB_PATH        = "chroma_db"
COLLECTION     = "book_chunks"
EMBED_MODEL    = "nomic-embed-text"   # ollama pull nomic-embed-text
CHUNK_SIZE     = 350   # words per chunk  (smaller = more precise retrieval)
CHUNK_OVERLAP  = 80    # words of overlap (prevents edge-of-chunk fact loss)
PROGRESS_FILE  = "index_progress.json"
# ─────────────────────────────────────────────────────────────────────────────


# ── Readers ───────────────────────────────────────────────────────────────────

def read_pdf(path: Path) -> str:
    doc = fitz.open(str(path))
    pages = [page.get_text("text") for page in doc]
    doc.close()
    return "\n\n".join(pages)


def read_epub(path: Path) -> str:
    book = epub.read_epub(str(path))
    parts = []
    for item in book.get_items():
        if item.get_type() == 9:           # ITEM_DOCUMENT
            html = item.get_content().decode("utf-8", errors="ignore")
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style"]):
                tag.decompose()
            parts.append(soup.get_text(separator="\n"))
    return "\n\n".join(parts)


def read_docx(path: Path) -> str:
    doc = docx.Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def read_json(path: Path) -> str:
    """Flatten JSON to readable text so facts inside are retrievable."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        data = json.load(f)

    def flatten(obj, depth=0) -> str:
        if isinstance(obj, dict):
            lines = []
            for k, v in obj.items():
                lines.append(f"{'  ' * depth}{k}: {flatten(v, depth+1)}")
            return "\n".join(lines)
        elif isinstance(obj, list):
            return "\n".join(flatten(i, depth) for i in obj)
        else:
            return str(obj)

    return flatten(data)


def read_html(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def read_csv(path: Path) -> str:
    """Convert CSV rows to natural-language sentences so they embed well."""
    import csv
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sentence = ". ".join(f"{k} is {v}" for k, v in row.items() if v.strip())
            rows.append(sentence)
    return "\n".join(rows)


def read_file(path: Path) -> str:
    ext = path.suffix.lower()
    try:
        if ext == ".pdf":   return read_pdf(path)
        if ext == ".epub":  return read_epub(path)
        if ext == ".docx":  return read_docx(path)
        if ext == ".json":  return read_json(path)
        if ext in (".html", ".htm"): return read_html(path)
        if ext == ".csv":   return read_csv(path)
        # .txt, .md and everything else
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"  ⚠️  Could not read {path.name}: {e}")
        return ""


# ── Chunking ─────────────────────────────────────────────────────────────────

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """
    Split text into overlapping word-level chunks.
    Tries to break at sentence boundaries inside each window.
    """
    # Normalise whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenise into words (keep punctuation attached)
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        window = " ".join(words[start:end])

        # Try to end at last sentence boundary in window
        match = list(re.finditer(r"[.!?]\s", window))
        if match and end < len(words):
            last = match[-1].end()
            window = window[:last].strip()

        if len(window.split()) >= 20:   # skip tiny fragments
            chunks.append(window)

        start += size - overlap          # slide with overlap

    return chunks


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed(text: str) -> list[float]:
    """Real semantic embedding via nomic-embed-text through Ollama."""
    resp = ollama.embeddings(model=EMBED_MODEL, prompt=text[:2000])
    return resp["embedding"]


# ── Progress tracking ─────────────────────────────────────────────────────────

def load_progress() -> set:
    if Path(PROGRESS_FILE).exists():
        return set(json.loads(Path(PROGRESS_FILE).read_text()))
    return set()


def save_progress(done: set):
    Path(PROGRESS_FILE).write_text(json.dumps(list(done)))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  BOOK INDEXER  —  semantic embeddings + ChromaDB")
    print("=" * 65)

    if not BOOKS_FOLDER.exists():
        BOOKS_FOLDER.mkdir()
        print(f"Created '{BOOKS_FOLDER}' — add your books there and re-run.")
        return

    SUPPORTED = {".pdf", ".epub", ".docx", ".txt", ".md",
                 ".json", ".html", ".htm", ".csv"}
    all_files = [f for f in BOOKS_FOLDER.iterdir()
                 if f.is_file() and f.suffix.lower() in SUPPORTED]

    if not all_files:
        print(f"No supported files found in '{BOOKS_FOLDER}'.")
        return

    # ── DB setup ──
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        col = client.get_collection(COLLECTION)
        print(f"Resuming — {col.count():,} chunks already stored.")
    except Exception:
        col = client.create_collection(
            name=COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )
        print("Created new collection.")

    done = load_progress()
    pending = [f for f in all_files if f.stem not in done]

    print(f"\nFiles found : {len(all_files)}")
    print(f"Already done: {len(done)}")
    print(f"To index    : {len(pending)}\n")

    if not pending:
        print("✅ Nothing new to index.")
        return

    total_chunks = 0
    t0 = time.time()

    for file_idx, fpath in enumerate(pending, 1):
        print(f"[{file_idx}/{len(pending)}] {fpath.name}")

        text = read_file(fpath)
        if len(text.strip()) < 50:
            print("  ⚠️  Skipping — too little text.")
            done.add(fpath.stem)
            save_progress(done)
            continue

        chunks = chunk_text(text)
        print(f"  {len(text):,} chars → {len(chunks)} chunks")

        ids, embeddings, documents, metadatas = [], [], [], []

        for chunk in tqdm(chunks, desc="  Embedding", unit="chunk", leave=False):
            chunk_id = f"{fpath.stem}_{hashlib.md5(chunk.encode()).hexdigest()[:12]}"

            # Skip if already in DB (safe resume)
            existing = col.get(ids=[chunk_id])
            if existing["ids"]:
                continue

            ids.append(chunk_id)
            embeddings.append(embed(chunk))
            documents.append(chunk)
            metadatas.append({
                "source": fpath.name,
                "book":   fpath.stem,
                "ext":    fpath.suffix.lower(),
            })

            # Batch-insert every 50 chunks to avoid memory spikes
            if len(ids) >= 50:
                col.add(embeddings=embeddings, documents=documents,
                        ids=ids, metadatas=metadatas)
                total_chunks += len(ids)
                ids, embeddings, documents, metadatas = [], [], [], []

        if ids:
            col.add(embeddings=embeddings, documents=documents,
                    ids=ids, metadatas=metadatas)
            total_chunks += len(ids)

        done.add(fpath.stem)
        save_progress(done)
        elapsed = time.time() - t0
        rate = total_chunks / elapsed if elapsed else 0
        print(f"  ✅ Done  |  total chunks so far: {col.count():,}  |  {rate:.1f} chunks/s")

    elapsed = time.time() - t0
    print("\n" + "=" * 65)
    print(f"✅  Indexing complete")
    print(f"   Chunks added this run : {total_chunks:,}")
    print(f"   Total in DB           : {col.count():,}")
    print(f"   Time                  : {elapsed/60:.1f} min")
    print("=" * 65)
    print("\nNext step → run:  python server.py")


if __name__ == "__main__":
    main()