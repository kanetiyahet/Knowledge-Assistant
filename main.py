from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import ollama

app = FastAPI(title="Gandhi Knowledge Assistant")

# === ADD CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SAME model as index_data.py
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

client = QdrantClient(path="qdrant_db", force_disable_check_same_thread=True)
collection_name = "gandhi_library"

class QueryRequest(BaseModel):
    question: str

class SourceInfo(BaseModel):
    book: str
    page: int
    snippet: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]

def build_prompt(question: str, retrieved_chunks: list) -> str:
    context = ""
    for i, chunk in enumerate(retrieved_chunks):
        context += (
            f"[Source {i+1}]\n"
            f"Book: {chunk.payload['book']}\n"
            f"Page: {chunk.payload['page']}\n"
            f"Text: {chunk.payload['text']}\n\n"
        )
    prompt = f"""You are a research assistant at the Gandhi Ashram Library. 
Use ONLY the following passages to answer the question. 
If the answer is not found, say exactly: 'Information not found in the library database.'
Always cite the Source number(s) you used.

{context}

Question: {question}
Answer with citations:"""
    return prompt

@app.post("/ask", response_model=QueryResponse)
async def ask_question(req: QueryRequest):
    # Embed the question
    question_embedding = model.encode([req.question]).tolist()[0]

    # Retrieve top 3 chunks
    search_result = client.query_points(
      collection_name=collection_name,
      query=question_embedding,
      limit=3
    ).points

    if not search_result:
        return QueryResponse(
            answer="Information not found in the library database.",
            sources=[]
        )

    # Generate answer with Ollama
    prompt = build_prompt(req.question, search_result)
    response = ollama.chat(model="qwen2.5:3b", messages=[
        {"role": "user", "content": prompt}
    ])
    answer = response["message"]["content"]

    # Prepare source citations
    sources = []
    for hit in search_result:
        sources.append(SourceInfo(
            book=hit.payload.get("book", ""),
            page=hit.payload.get("page", 0),
            snippet=hit.payload.get("text", "")[:200] + "..."
        ))

    return QueryResponse(answer=answer, sources=sources)