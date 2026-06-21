import os, fitz, chromadb, json
import ollama
from sentence_transformers import SentenceTransformer

BOOKS_FOLDER = "books"
DB_PATH = "facts_db"

# ===== DON'T DELETE OLD FACTS =====
chroma_client = chromadb.PersistentClient(path=DB_PATH)

# Check if collection exists
try:
    collection = chroma_client.get_collection("facts")
    existing_count = collection.count()
    print(f"📂 Found existing database with {existing_count} facts")
except:
    # Create new only if doesn't exist
    collection = chroma_client.create_collection("facts", metadata={"hnsw:space": "cosine"})
    existing_count = 0
    print("📂 Created new database")

model = SentenceTransformer("all-MiniLM-L6-v2")

# Get already processed books
existing_books = set()
if existing_count > 0:
    all_meta = collection.get()['metadatas']
    existing_books = set(m['book'] for m in all_meta)
    print(f"📚 Already processed: {existing_books}")

pdf_files = [f for f in os.listdir(BOOKS_FOLDER) if f.endswith(".pdf")]

new_facts = 0

for pdf_file in pdf_files:
    book_name = os.path.splitext(pdf_file)[0]
    
    # 🔥 SKIP ALREADY PROCESSED BOOKS
    if book_name in existing_books:
        print(f"⏭️ Skipping (already done): {book_name}")
        continue
    
    pdf_path = os.path.join(BOOKS_FOLDER, pdf_file)
    if not os.path.exists(pdf_path):
        print(f"⚠️ Missing file: {pdf_file}")
        continue
    
    doc = fitz.open(pdf_path)
    print(f"\n📖 Processing: {book_name} ({len(doc)} pages)")
    
    for page_num in range(0, len(doc), 5):
        text = ""
        for p in range(page_num, min(page_num+5, len(doc))):
            text += doc[p].get_text("text") + "\n"
        
        if len(text.strip()) < 100:
            continue
        
        prompt = f"""Extract ALL key facts from this text. 
For each fact, write one clear sentence.
Include: who, what, where, when, why.
Focus on: birth, death, events, places, people, dates.

Text: {text[:2000]}

Facts (one per line):"""
        
        try:
            r = ollama.chat(model="qwen2.5:3b", messages=[{"role":"user","content":prompt}])
            facts = [f.strip() for f in r["message"]["content"].strip().split('\n') 
                     if f.strip() and len(f) > 20]
        except:
            continue
        
        for fact in facts:
            embedding = model.encode([fact]).tolist()[0]
            collection.add(
                embeddings=[embedding],
                documents=[fact],
                ids=[f"fact_{book_name}_{page_num}_{hash(fact)%100000}"],
                metadatas=[{"book": book_name, "page": page_num+1, "fact": fact}]
            )
            new_facts += 1
        
        print(f"  Pages {page_num+1}-{min(page_num+5, len(doc))}: {len(facts)} facts")
    
    doc.close()

print(f"\n{'='*50}")
print(f"✅ DONE!")
print(f"   New facts added: {new_facts}")
print(f"   Total facts: {collection.count()}")
print(f"{'='*50}")