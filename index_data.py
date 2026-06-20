import os, fitz, re, chromadb
from sentence_transformers import SentenceTransformer

BOOKS_FOLDER = "books"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
DB_PATH = "chroma_db"
COLLECTION_NAME = "book_chunks"

def semantic_chunk(text, max_words=200, overlap=50):
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

pdf_files = [f for f in os.listdir(BOOKS_FOLDER) if f.lower().endswith(".pdf")]
print(f"Found {len(pdf_files)} PDF(s): {pdf_files}")

chroma_client = chromadb.PersistentClient(path=DB_PATH)
try: chroma_client.delete_collection(COLLECTION_NAME)
except: pass
collection = chroma_client.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

model = SentenceTransformer(MODEL_NAME)
chunk_id = 0

for pdf_file in pdf_files:
    pdf_path = os.path.join(BOOKS_FOLDER, pdf_file)
    doc = fitz.open(pdf_path)
    book_name = os.path.splitext(pdf_file)[0]
    print(f"Processing: {book_name} ({len(doc)} pages)")
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        if not text.strip(): continue
        
        for chunk_text in semantic_chunk(text):
            if len(chunk_text.split()) < 10: continue
            chapter = ""
            m = re.search(r'(?:Chapter|CHAPTER)\s*[IVX0-9]+', chunk_text)
            if m: chapter = m.group(0)
            
            embedding = model.encode([chunk_text]).tolist()[0]
            collection.add(
                embeddings=[embedding],
                documents=[chunk_text],
                ids=[f"chunk_{chunk_id}"],
                metadatas=[{"book": book_name, "chapter": chapter, "page": page_num + 1, "text": chunk_text}]
            )
            chunk_id += 1
    doc.close()
    print(f"  -> {book_name} done")

print(f"\n✅ {chunk_id} chunks indexed in '{DB_PATH}'")