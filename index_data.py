import os, fitz, re, chromadb
from sentence_transformers import SentenceTransformer

BOOKS_FOLDER = "books"
MODEL_NAME = "all-MiniLM-L6-v2"  # Fast, good enough
DB_PATH = "chroma_db"
COLLECTION_NAME = "book_chunks"

def semantic_chunk(text, max_words=150, overlap=30):
    """Smart chunking with overlap"""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks, current_chunk, current_words = [], [], 0
    for para in paragraphs:
        para_words = len(para.split())
        if current_words + para_words > max_words and current_chunk:
            chunks.append(' '.join(current_chunk))
            overlap_text = ' '.join(current_chunk[-max(1, overlap//10):])
            current_chunk = [overlap_text] if overlap_text else []
            current_words = len(overlap_text.split()) if overlap_text else 0
        current_chunk.append(para)
        current_words += para_words
    if current_chunk: chunks.append(' '.join(current_chunk))
    return chunks

print("Setting up ChromaDB...")
chroma_client = chromadb.PersistentClient(path=DB_PATH)
try: chroma_client.delete_collection(COLLECTION_NAME)
except: pass
collection = chroma_client.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

print(f"Loading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

pdf_files = [f for f in os.listdir(BOOKS_FOLDER) if f.lower().endswith(".pdf")]
print(f"Found {len(pdf_files)} PDF(s)")

chunk_id = 0
for pdf_file in pdf_files:
    doc = fitz.open(os.path.join(BOOKS_FOLDER, pdf_file))
    book_name = os.path.splitext(pdf_file)[0]
    print(f"Processing: {book_name} ({len(doc)} pages)")
    
    for page_num in range(len(doc)):
        text = doc.load_page(page_num).get_text("text")
        if not text.strip(): continue
        
        for chunk_text in semantic_chunk(text):
            if len(chunk_text.split()) < 10: continue
            
            embedding = model.encode([chunk_text]).tolist()[0]
            collection.add(
                embeddings=[embedding],
                documents=[chunk_text],
                ids=[f"chunk_{chunk_id}"],
                metadatas=[{"book": book_name, "page": page_num + 1, "text": chunk_text}]
            )
            chunk_id += 1
    doc.close()

print(f"\n✅ {chunk_id} chunks indexed!")