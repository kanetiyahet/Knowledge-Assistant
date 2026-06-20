import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
import ollama

app = FastAPI(title="Book Q&A Assistant")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DB_PATH = "chroma_db"
COLLECTION_NAME = "book_chunks"

print("Loading...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_collection(COLLECTION_NAME)
print(f"✅ {collection.count()} chunks ready!")

class QueryRequest(BaseModel): question: str
class SourceInfo(BaseModel): book: str; page: int; snippet: str
class QueryResponse(BaseModel): answer: str; sources: list[SourceInfo]

def extract_answer(metadatas, question):
    if not metadatas:
        return "Information not found in the library database."
    
    ctx = ""; srcs = []
    for i, m in enumerate(metadatas[:3]):
        ctx += f"[{i+1}] {m.get('book','?')}, Page {m.get('page',0)}:\n{m.get('text','')[:300]}\n\n"
        srcs.append(f"📖 {m.get('book','?')} (Page {m.get('page',0)})")
    
    try:
        r = ollama.chat(model="qwen2.5:3b", messages=[{"role":"user","content":f"Answer in 2-3 sentences.\n\n{ctx}\nQuestion: {question}\nAnswer:"}])
        ans = r["message"]["content"].strip()
    except:
        ans = metadatas[0].get('text','')[:300]
    
    return ans + f"\n\n📚 Sources: {' | '.join(srcs)}"

@app.post("/ask", response_model=QueryResponse)
async def ask(req: QueryRequest):
    q = req.question.strip()
    print(f"\n📝 Q: {q[:80]}")
    
    emb = embedding_model.encode([q]).tolist()[0]
    res = collection.query(query_embeddings=[emb], n_results=3)
    metas = res['metadatas'][0] if res['metadatas'] else []
    
    ans = extract_answer(metas, q)
    srcs = [SourceInfo(book=m.get('book',''), page=m.get('page',0), snippet=m.get('text','')[:200]+"...") for m in metas]
    return QueryResponse(answer=ans, sources=srcs)

@app.get("/")
async def root(): return {"status":"running", "db":"ChromaDB", "chunks":collection.count()}