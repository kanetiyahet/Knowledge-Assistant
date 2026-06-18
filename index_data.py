import os
import fitz  # PyMuPDF
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# ===== CONFIGURATION =====
BOOKS_FOLDER = "books"
MODEL_NAME = "all-MiniLM-L6-v2"         # or "all-MiniLM-L6-v2" for speed
QDRANT_PATH = "qdrant_db"
COLLECTION_NAME = "gandhi_library"

# ===== LOAD ALL PDFS =====
pdf_files = [f for f in os.listdir(BOOKS_FOLDER) if f.lower().endswith(".pdf")]
if not pdf_files:
    raise FileNotFoundError(f"No PDF files found in '{BOOKS_FOLDER}' folder.")

print(f"Found {len(pdf_files)} PDF(s): {pdf_files}")

chunks = []

for pdf_file in pdf_files:
    pdf_path = os.path.join(BOOKS_FOLDER, pdf_file)
    doc = fitz.open(pdf_path)
    book_name = os.path.splitext(pdf_file)[0]   # e.g., "Hind Swaraj"

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")   # extract plain text
        if not text.strip():
            continue

        # Split page text into paragraphs (simple heuristic: blank lines)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        for para in paragraphs:
            # Ignore very short lines (headers, page numbers)
            if len(para.split()) < 8:
                continue
            chunks.append({
                "text": para,
                "book": book_name,
                "chapter": "",          # we'll add chapter detection later
                "page": page_num + 1    # 1‑based page number
            })

    doc.close()

print(f"Created {len(chunks)} text chunks from all PDFs.")

# ===== EMBEDDING =====
print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)
embeddings = model.encode([c["text"] for c in chunks], show_progress_bar=True)

# ===== STORE IN QDRANT (LOCAL MODE) =====
client = QdrantClient(path=QDRANT_PATH)

# Recreate collection if already exists
if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.COSINE)
)

points = [
    PointStruct(id=i, vector=embeddings[i].tolist(), payload=chunks[i])
    for i in range(len(chunks))
]

# Insert in batches (to avoid memory issues with huge libraries)
BATCH_SIZE = 500
for i in range(0, len(points), BATCH_SIZE):
    batch = points[i:i+BATCH_SIZE]
    client.upsert(collection_name=COLLECTION_NAME, points=batch)

print(f"Indexing complete. {len(chunks)} chunks stored in '{QDRANT_PATH}'.")