import os, json, numpy as np, re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
import ollama

app = FastAPI(title="Smart Chunk RAG")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

print("Loading...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_collection("book_chunks")
print(f"✅ {collection.count()} chunks ready!")

class QueryRequest(BaseModel): question: str

def quick_fix(text):
    fixes = {'ghandhi':'Gandhi','ghandi':'Gandhi','ghndhi':'Gandhi','pleace':'place','wat':'what','wer':'where'}
    for w, c in fixes.items():
        if w in text.lower(): text = re.sub(w, c, text, flags=re.IGNORECASE)
    return text

def add_search_tags(query):
    """Enhance query with topic tags"""
    lower = query.lower()
    tags = []
    if any(w in lower for w in ['born','birth','birthplace']): tags.append('BIRTH_INFO')
    if any(w in lower for w in ['die','death','assassinated']): tags.append('DEATH_INFO')
    if 'satyagraha' in lower: tags.append('SATYAGRAHA')
    if 'ahimsa' in lower or 'non-violen' in lower: tags.append('AHIMSA')
    if 'dandi' in lower or 'salt march' in lower: tags.append('DANDI_MARCH')
    if 'south africa' in lower: tags.append('SOUTH_AFRICA')
    if tags:
        return query + " [TAGS: " + " ".join(tags) + "]"
    return query

@app.post("/ask")
async def ask(req: QueryRequest):
    q = quick_fix(req.question.strip())
    enhanced_q = add_search_tags(q)
    print(f"\n📝 Q: {q}")
    print(f"🔍 Enhanced: {enhanced_q}")
    
    # Search chunks
    q_emb = embedding_model.encode([enhanced_q]).tolist()[0]
    results = collection.query(query_embeddings=[q_emb], n_results=10)
    
    chunks = []
    for i, doc in enumerate(results['documents'][0]):
        chunks.append({
            'text': doc,
            'metadata': results['metadatas'][0][i]
        })
    
    # Rerank
    pairs = [[enhanced_q, c['text'][:500]] for c in chunks]
    scores = reranker.predict(pairs)
    scored = list(zip(chunks, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    top = [c for c, s in scored[:3]]
    
    # Build context
    context = ""
    sources = []
    for i, c in enumerate(top):
        book = c['metadata'].get('book', 'Unknown')
        page = c['metadata'].get('page', 0)
        text = c['text'][:500]
        context += f"[Source {i+1}] {book}, Page {page}:\n{text}\n\n"
        sources.append(f"{book} (Page {page})")
    
    print(f"📊 Top: {sources}")
    
    # Generate answer
    prompt = f"""Answer the question using ONLY the sources below.
Be direct, specific, and factual.
If the answer is not in the sources, say "Not found in the books."

Sources:
{context}

Question: {q}
Answer:"""
    
    try:
        r = ollama.chat(model="qwen2.5:3b", messages=[{"role":"user","content":prompt}])
        answer = r["message"]["content"].strip()
    except:
        answer = "Error processing."
    
    answer += f"\n\n📚 Sources: {' | '.join(sources)}"
    
    source_objects = [
        {"book": c['metadata'].get('book','?'), "page": c['metadata'].get('page',0), "snippet": c['text'][:150]} 
        for c in top
    ]
    
    return {"answer": answer, "sources": source_objects}

@app.get("/")
async def root(): 
    return {"status": "Smart Chunk RAG Active", "chunks": collection.count()}