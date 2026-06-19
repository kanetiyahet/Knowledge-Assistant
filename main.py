# ============ IMPORTS ============
import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import ollama
import requests
from dotenv import load_dotenv

# Explicit path so .env is found regardless of where you run uvicorn from
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

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
SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "").strip()
SARVAM_URL = "https://api.sarvam.ai/translate"

if not SARVAM_API_KEY:
    print("⚠️  WARNING: SARVAM_API_KEY is missing or empty. "
          "Check that .env exists next to this file and has no quotes around the key.")
else:
    print(f"✅ Sarvam key loaded (length: {len(SARVAM_API_KEY)})")

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
def detect_language(text: str) -> str:
    gujarati_chars = sum(1 for c in text if '\u0A80' <= c <= '\u0AFF')
    hindi_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    total = max(len(text), 1)

    if gujarati_chars / total > 0.2:
        return "gujarati"
    elif hindi_chars / total > 0.2:
        return "hindi"
    return "english"

# ============ SARVAM TRANSLATION ============
def translate_with_sarvam(text: str, target_lang: str) -> str | None:
    """Returns translated text, or None on any failure (caller decides fallback)."""
    if not SARVAM_API_KEY:
        print("DEBUG: No Sarvam key set, skipping translation.")
        return None

    lang_codes = {"hindi": "hi-IN", "gujarati": "gu-IN"}
    target_code = lang_codes.get(target_lang)
    if not target_code:
        print(f"DEBUG: Unknown target language '{target_lang}'")
        return None

    # Sarvam has an input length cap — truncate safely to avoid silent failure
    safe_text = text[:990]

    try:
        resp = requests.post(
            SARVAM_URL,
            json={
                "input": safe_text,
                "source_language_code": "en-IN",
                "target_language_code": target_code,
                "mode": "formal",
            },
            headers={
                "api-subscription-key": SARVAM_API_KEY,
                "Content-Type": "application/json",
            },
            timeout=15,
        )

        print(f"DEBUG: Sarvam status={resp.status_code}")

        if resp.status_code != 200:
            print(f"DEBUG: Sarvam error body: {resp.text}")
            return None

        translated = resp.json().get("translated_text", "").strip()

        if len(translated) < 5:
            print(f"DEBUG: Sarvam returned suspiciously short text: '{translated}'")
            return None

        return translated

    except requests.exceptions.Timeout:
        print("DEBUG: Sarvam request timed out")
        return None
    except Exception as e:
        print(f"DEBUG: Sarvam exception: {e}")
        return None

# ============ EXTRACT ANSWER ============
def extract_answer(chunks, lang: str, question: str) -> str:
    print(f"DEBUG: lang={lang}, question={question[:50]}")

    if not chunks:
        not_found = {
            "english": "Information not found in the library database.",
            "hindi": "पुस्तकालय डेटाबेस में जानकारी नहीं मिली।",
            "gujarati": "લાઇબ્રેરી ડેટાબેઝમાં માહિતી મળી નથી.",
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

    prompt = f"""Answer in 2-3 clear English sentences. Use ONLY the sources below.

{context}

Question: {question}
Answer:"""

    try:
        response = ollama.chat(
            model="qwen2.5:3b",
            messages=[{"role": "user", "content": prompt}],
        )
        english_answer = response["message"]["content"].strip()
        english_answer = english_answer.replace("Answer:", "").strip()
    except Exception as e:
        print(f"DEBUG: Ollama exception: {e}")
        english_answer = chunks[0].payload.get('text', '')[:300]

    source_text = f"\n\n📚 Sources: {' | '.join(sources_list)}"

    if lang == "english":
        return english_answer + source_text

    print(f"DEBUG: Translating to {lang}...")
    translated = translate_with_sarvam(english_answer, lang)

    if translated:
        return translated + source_text

    print(f"DEBUG: Translation to {lang} failed — falling back to English")
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
        limit=3,
    ).points

    answer = extract_answer(search_result, lang, original_question)
    print(f"✅ Answer: {answer[:150]}...")

    sources = []
    for hit in (search_result or []):
        sources.append(SourceInfo(
            book=hit.payload.get("book", ""),
            page=hit.payload.get("page", 0),
            snippet=hit.payload.get("text", "")[:200] + "...",
        ))

    return QueryResponse(answer=answer, sources=sources)

@app.get("/")
async def root():
    return {
        "status": "running",
        "languages": ["english", "hindi", "gujarati"],
        "translation": "Sarvam AI",
        "sarvam_key_loaded": bool(SARVAM_API_KEY),
    }