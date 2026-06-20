import os, re, chromadb
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

BOOKS_FOLDER = "books"
DB_PATH = "chroma_db"
MODEL_NAME = "all-MiniLM-L6-v2"

print("🚀 EPUB Indexing System")
print("=" * 50)

# Setup ChromaDB
chroma_client = chromadb.PersistentClient(path=DB_PATH)
try: chroma_client.delete_collection("book_chunks")
except: pass
collection = chroma_client.create_collection("book_chunks", metadata={"hnsw:space": "cosine"})

# Load model
model = SentenceTransformer(MODEL_NAME)

# Find EPUB files
epub_files = [f for f in os.listdir(BOOKS_FOLDER) if f.endswith('.epub')]
print(f"📚 Found {len(epub_files)} EPUB file(s)")

chunk_id = 0

for epub_file in epub_files:
    book_name = os.path.splitext(epub_file)[0]
    filepath = os.path.join(BOOKS_FOLDER, epub_file)
    if not os.path.exists(filepath):
    	print(f"  ⚠️ File not found, skipping: {epub_file}")
    	continue
    print(f"\n📖 Processing: {book_name}")
    
    # Read EPUB
    book = epub.read_epub(filepath)
    
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            text = soup.get_text(separator=' ')
            
            # Clean text
            text = re.sub(r'\s+', ' ', text).strip()
            if len(text) < 100:
                continue
            
            # Split into chunks
            sentences = re.split(r'(?<=[.!?])\s+', text)
            current_chunk = []
            current_words = 0
            
            for s in sentences:
                words = len(s.split())
                if current_words + words > 100 and current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    embedding = model.encode([chunk_text]).tolist()[0]
                    collection.add(
                        embeddings=[embedding],
                        documents=[chunk_text],
                        ids=[f"c_{chunk_id}"],
                        metadatas=[{"book": book_name, "chapter": item.get_name(), "text": chunk_text}]
                    )
                    chunk_id += 1
                    current_chunk = []
                    current_words = 0
                current_chunk.append(s)
                current_words += words
            
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                embedding = model.encode([chunk_text]).tolist()[0]
                collection.add(
                    embeddings=[embedding],
                    documents=[chunk_text],
                    ids=[f"c_{chunk_id}"],
                    metadatas=[{"book": book_name, "chapter": item.get_name(), "text": chunk_text}]
                )
                chunk_id += 1
    
    print(f"  ✅ Done")

print(f"\n{'='*50}")
print(f"🎉 Total chunks: {chunk_id}")
print(f"📂 Database: {DB_PATH}")