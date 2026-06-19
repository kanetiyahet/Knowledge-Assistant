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

SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "") # ← PUT YOUR KEY HERE
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

# ============ SARVAM TRANSLATION ============
def translate_sarvam(text, source_lang, target_lang):
    if source_lang == target_lang:
        return text
    
    lang_codes = {
        "english": "en-IN",
        "hindi": "hi-IN",
        "gujarati": "gu-IN"
    }
    
    payload = {
        "input": text,
        "source_language_code": lang_codes.get(source_lang, "en-IN"),
        "target_language_code": lang_codes.get(target_lang, "en-IN"),
        "mode": "formal"
    }
    
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(SARVAM_URL, json=payload, headers=headers, timeout=10)
        if response.status_code == 200:
            result = response.json().get("translated_text", "")
            return result if result else text
    except:
        pass
    return text

# ============ SIMPLE ANSWER FROM CHUNKS ============
def extract_answer(chunks, lang):
    """Extract best answer directly from chunks - NO LLM needed"""
    
    if not chunks:
        not_found = {
            "english": "Information not found in the library database.",
            "hindi": "पुस्तकालय डेटाबेस में यह जानकारी नहीं मिली।",
            "gujarati": "લાઇબ્રેરી ડેટાબેઝમાં આ માહિતી મળી નથી."
        }
        return not_found.get(lang, not_found["english"])
    
    # Take the best matching chunk
    best = chunks[0]
    book = best.payload.get('book', 'Unknown')
    page = best.payload.get('page', 0)
    text = best.payload.get('text', '')[:400]
    
    # Build English answer
    english_answer = f"📖 From {book} (Page {page}):\n\n\"{text}...\""
    
    # Translate to user's language if needed
    if lang == "english":
        return english_answer
    elif lang in ["hindi", "gujarati"]:
        translated = translate_sarvam(english_answer, "english", lang)
        return translated if translated else english_answer
    
    return english_answer

# ============ MAIN ENDPOINT ============
@app.post("/ask", response_model=QueryResponse)
async def ask_question(req: QueryRequest):
    original_question = req.question.strip()
    
    # Detect language
    lang = detect_language(original_question)
    print(f"\n🌐 Language: {lang} | Q: {original_question[:80]}")
    
    # Search in Qdrant (use original question)
    question_embedding = model.encode([original_question]).tolist()[0]
    search_result = client.query_points(
        collection_name=collection_name,
        query=question_embedding,
        limit=3
    ).points
    
    # Extract answer directly from chunks
    answer = extract_answer(search_result, lang)
    print(f"✅ Answer: {answer[:150]}...")
    
    # Prepare sources
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
        "method": "Direct chunk extraction + Sarvam translation",
        "languages": ["english", "hindi", "gujarati"]
    }