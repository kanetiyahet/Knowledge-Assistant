# ============ IMPORTS ============
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import ollama
import requests

# ============ APP SETUP ============
app = FastAPI(title="Gandhi Knowledge Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ CONFIGURATION ============
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "your_key_here")
SARVAM_URL = "https://api.sarvam.ai/translate"

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

# ============ LANGUAGE DETECTION ============
def detect_language(text):
    gujarati_chars = sum(1 for c in text if '\u0A80' <= c <= '\u0AFF')
    hindi_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    
    if gujarati_chars > len(text) * 0.2:
        return "gujarati"
    elif hindi_chars > len(text) * 0.2:
        return "hindi"
    return "english"

# ============ EXTRACT ANSWER ============
def extract_answer(chunks, lang, question):
    """English answer + Sarvam translation"""
    print(f"DEBUG: lang={lang}, question={question[:50]}")
    
    if not chunks:
        not_found = {
            "english": "Information not found in the library database.",
            "hindi": "पुस्तकालय डेटाबेस में जानकारी नहीं मिली।",
            "gujarati": "લાઇબ્રેરી ડેટાબેઝમાં માહિતી મળી નથી."
        }
        return not_found.get(lang, not_found["english"])
    
    context = ""
    sources_list = []
    for i, chunk in enumerate(chunks[:2]):
        book = chunk.payload.get('book', 'Unknown')
        page = chunk.payload.get('page', 0)
        text = chunk.payload.get('text', '')[:300]
        context += f"[{i+1}] {book}, Page {page}:\n{text}\n\n"
        sources_list.append(f"📖 {book} (Page {page})")
    
    # Generate English answer
    prompt = f"""Answer in 2-3 clear English sentences. Use ONLY the sources below.

{context}

Question: {question}
Answer:"""
    
    try:
        response = ollama.chat(model="qwen2.5:3b", messages=[{"role": "user", "content": prompt}])
        english_answer = response["message"]["content"].strip()
        english_answer = english_answer.replace("Answer:", "").strip()
    except:
        english_answer = chunks[0].payload.get('text', '')[:300]
    
    source_text = f"\n\n📚 Sources: {' | '.join(sources_list)}"
    
    # Return English directly
    if lang == "english":
        return english_answer + source_text
    
    # Translate via Sarvam
    lang_codes = {"hindi": "hi-IN", "gujarati": "gu-IN"}
    
    try:
        resp = requests.post(SARVAM_URL, json={
            "input": english_answer,
            "source_language_code": "en-IN",
            "target_language_code": lang_codes.get(lang, "hi-IN"),
            "mode": "formal"
        }, headers={
            "api-subscription-key": SARVAM_API_KEY,
            "Content-Type": "application/json"
        }, timeout=15)
        
        if resp.status_code == 200:
            translated = resp.json().get("translated_text", "")
            if translated and len(translated) > 10:
                return translated + source_text
    except:
        pass
    
    # Fallback: English
    return english_answer + source_text

# ============ MAIN ENDPOINT ============
@app.post("/ask", response_model=QueryResponse)
async def ask_question(req: QueryRequest):
    original_question = req.question.strip()
    
    lang = detect_language(original_question)
    print(f"\n🌐 Language: {lang} | Q: {original_question[:80]}")
    
    question_embedding = model.encode([original_question]).tolist()[0]
    search_result = client.query_points(
        collection_name=collection_name,
        query=question_embedding,
        limit=3
    ).points
    
    answer = extract_answer(search_result, lang, original_question)
    print(f"✅ Answer: {answer[:150]}...")
    
    sources = []
    for hit in (search_result or []):
        sources.append(SourceInfo(
            book=hit.payload.get("book", ""),
            page=hit.payload.get("page", 0),
            snippet=hit.payload.get("text", "")[:200] + "..."
        ))
    
    return QueryResponse(answer=answer, sources=sources)

@app.get("/")
async def root():
    return {
        "status": "running",
        "languages": ["english", "hindi", "gujarati"],
        "translation": "Sarvam AI"
    }