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
def extract_answer(chunks, lang, question):
    """Use LLM to generate clean summarized answer"""
    
    if not chunks:
        not_found = {
            "english": "Information not found in the library database.",
            "hindi": "पुस्तकालय डेटाबेस में यह जानकारी नहीं मिली।",
            "gujarati": "લાઇબ્રેરી ડેટાબેઝમાં આ માહિતી મળી નથી."
        }
        return not_found.get(lang, not_found["english"])
    
    # Build context from top chunks
    context = ""
    sources_list = []
    for i, chunk in enumerate(chunks[:2]):
        book = chunk.payload.get('book', 'Unknown')
        page = chunk.payload.get('page', 0)
        text = chunk.payload.get('text', '')[:400]
        context += f"[{i+1}] {book}, Page {page}:\n{text}\n\n"
        sources_list.append(f"📖 {book} (Page {page})")
    
    # Language instruction
    lang_instruction = {
        "english": "Answer in English in 2-3 sentences.",
        "hindi": "हिंदी में 2-3 वाक्यों में उत्तर दें।",
        "gujarati": "ગુજરાતીમાં 2-3 વાક્યોમાં જવાબ આપો."
    }
    
    prompt = f"""{lang_instruction.get(lang, 'Answer in English.')}
Use ONLY the context below. Be accurate and mention the source.

Context:
{context}

Question: {question}
Answer:"""
    
    try:
        response = ollama.chat(model="qwen2.5:3b", messages=[{"role": "user", "content": prompt}])
        answer = response["message"]["content"].strip()
    except:
        # Fallback
        text = chunks[0].payload.get('text', '')[:300]
        book = chunks[0].payload.get('book', '')
        page = chunks[0].payload.get('page', 0)
        answer = f"From {book} (Page {page}):\n\"{text}...\""
    
    # Add sources
    answer += f"\n\n📚 Sources: {' | '.join(sources_list)}"
    
    return answer

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
    answer = extract_answer(search_result, lang, original_question)
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