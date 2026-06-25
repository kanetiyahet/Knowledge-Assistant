"""
server.py
---------
Q&A server over your indexed books.

Features:
  - Real vector search (nomic-embed-text via Ollama)
  - Keyword reranking on top of semantic results
  - Retrieves top-15, reranks, feeds top-6 to LLM
  - Cites which book/file the answer came from
  - Clean, responsive UI

Run:
    python server.py
    → http://localhost:8000
"""

import re
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import chromadb
import ollama
import uvicorn


# ─── CONFIG ──────────────────────────────────────────────────────────────────
DB_PATH      = "chroma_db"
COLLECTION   = "book_chunks"
EMBED_MODEL  = "nomic-embed-text"
CHAT_MODEL   = "qwen2.5:3b"       # swap for any model you have pulled
RETRIEVE_N   = 15                  # how many chunks to pull from vector DB
FINAL_N      = 6                   # how many chunks to send to LLM after rerank
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load DB ──────────────────────────────────────────────────────────────────
print("Loading database …")
client = chromadb.PersistentClient(path=DB_PATH)
col    = client.get_collection(COLLECTION)
print(f"✅  {col.count():,} chunks ready")


# ── Helpers ───────────────────────────────────────────────────────────────────

def embed(text: str) -> list[float]:
    return ollama.embeddings(model=EMBED_MODEL, prompt=text[:2000])["embedding"]


def keyword_score(text: str, query: str) -> float:
    """Simple overlap score, normalised 0-1."""
    q_words = set(re.sub(r"[^\w\s]", "", query.lower()).split())
    t_words = set(re.sub(r"[^\w\s]", "", text.lower()).split())
    if not q_words:
        return 0.0
    return len(q_words & t_words) / len(q_words)


def rerank(docs, metas, distances, query: str, top_n: int = FINAL_N):
    """
    Combine semantic similarity (cosine distance) with keyword overlap.
    Semantic similarity carries 70% weight, keyword 30%.
    """
    scored = []
    for doc, meta, dist in zip(docs, metas, distances):
        semantic  = 1.0 - dist                       # higher = more similar
        keyword   = keyword_score(doc, query)
        combined  = 0.70 * semantic + 0.30 * keyword
        scored.append((doc, meta, combined))

    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:top_n]


def build_context(ranked: list) -> tuple[str, list[str]]:
    """Return (context_string, source_list)."""
    parts   = []
    sources = []
    for i, (doc, meta, score) in enumerate(ranked, 1):
        src = meta.get("source", "unknown")
        parts.append(f"[{i}] ({src})\n{doc}")
        if src not in sources:
            sources.append(src)
    return "\n\n".join(parts), sources


def ask_llm(context: str, query: str) -> str:
    prompt = f"""You are an expert assistant. Answer the question below using ONLY
the context provided. Be detailed and accurate.

If the context does not contain enough information to answer, say:
"The available books don't cover this topic in enough detail."

Never invent information that is not in the context.

--- CONTEXT START ---
{context}
--- CONTEXT END ---

Question: {query}

Answer:"""

    resp = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1},
    )
    return resp["message"]["content"].strip()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/ask")
async def ask(request: Request):
    try:
        body  = await request.json()
        query = body.get("query", "").strip()
        if not query:
            return JSONResponse({"error": "Empty query"}, status_code=400)

        print(f"\n🔍  {query}")

        # 1. Semantic search
        q_emb = embed(query)
        results = col.query(
            query_embeddings=[q_emb],
            n_results=min(RETRIEVE_N, col.count()),
            include=["documents", "metadatas", "distances"],
        )
        docs      = results["documents"][0]
        metas     = results["metadatas"][0]
        distances = results["distances"][0]

        if not docs:
            return {"answer": "No relevant information found in the books.", "sources": []}

        # 2. Rerank
        ranked = rerank(docs, metas, distances, query)

        # 3. Build context
        context, sources = build_context(ranked)

        # 4. Generate answer
        answer = ask_llm(context, query)

        print(f"✅  Answer generated | sources: {sources}")
        return {
            "answer":  answer,
            "sources": [{"source": s} for s in sources],
        }

    except Exception as e:
        print(f"❌  Error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/stats")
async def stats():
    return {"total_chunks": col.count(), "collection": COLLECTION}


@app.get("/", response_class=HTMLResponse)
async def home():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Book Q&A</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'Segoe UI', Arial, sans-serif;
    background: #0f172a;
    color: #e2e8f0;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 40px 16px;
  }

  .card {
    background: #1e293b;
    border-radius: 16px;
    padding: 36px;
    width: 100%;
    max-width: 780px;
    box-shadow: 0 8px 40px rgba(0,0,0,0.5);
  }

  h1 { font-size: 1.8rem; font-weight: 700; color: #f8fafc; }
  .sub { color: #94a3b8; margin-top: 4px; font-size: 0.9rem; }

  .input-row {
    display: flex;
    gap: 10px;
    margin-top: 28px;
  }

  textarea {
    flex: 1;
    padding: 14px 16px;
    background: #0f172a;
    border: 1.5px solid #334155;
    border-radius: 10px;
    color: #e2e8f0;
    font-size: 15px;
    resize: none;
    min-height: 56px;
    max-height: 160px;
    overflow-y: auto;
    line-height: 1.5;
    outline: none;
    transition: border-color 0.2s;
  }
  textarea:focus { border-color: #6366f1; }
  textarea::placeholder { color: #475569; }

  button {
    padding: 14px 24px;
    background: #6366f1;
    color: #fff;
    border: none;
    border-radius: 10px;
    font-size: 15px;
    font-weight: 600;
    cursor: pointer;
    align-self: flex-end;
    transition: background 0.2s, transform 0.1s;
    white-space: nowrap;
  }
  button:hover:not(:disabled) { background: #4f46e5; transform: translateY(-1px); }
  button:disabled { opacity: 0.55; cursor: not-allowed; }

  .chips {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 14px;
  }
  .chip {
    background: #1e3a5f;
    color: #93c5fd;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 12px;
    cursor: pointer;
    border: 1px solid #1d4ed8;
    transition: background 0.15s;
  }
  .chip:hover { background: #1d4ed8; }

  .answer-card {
    margin-top: 28px;
    background: #0f172a;
    border-radius: 12px;
    padding: 24px;
    display: none;
    border: 1px solid #1e293b;
    animation: fadeUp 0.35s ease;
  }
  .answer-card.show { display: block; }

  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  .answer-text {
    white-space: pre-wrap;
    line-height: 1.75;
    color: #cbd5e1;
    font-size: 15px;
  }

  .sources {
    margin-top: 18px;
    padding-top: 14px;
    border-top: 1px solid #1e293b;
  }
  .sources-label { font-size: 12px; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }
  .source-pill {
    display: inline-block;
    background: #1e293b;
    border: 1px solid #334155;
    color: #94a3b8;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 12px;
    margin: 6px 4px 0 0;
  }

  .spinner {
    width: 18px; height: 18px;
    border: 3px solid rgba(255,255,255,0.2);
    border-top-color: #fff;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    display: inline-block;
    vertical-align: middle;
    margin-right: 8px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  .error { color: #f87171; }
</style>
</head>
<body>
<div class="card">
  <h1>📚 Book Q&amp;A</h1>
  <p class="sub" id="stats-label">Loading stats…</p>

  <div class="input-row">
    <textarea id="q" placeholder="Ask anything about your books…" rows="2"
      onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();ask();}"></textarea>
    <button id="btn" onclick="ask()">Ask</button>
  </div>

  <div class="chips" id="chips"></div>

  <div class="answer-card" id="answer-card">
    <div class="answer-text" id="answer-text"></div>
    <div class="sources" id="sources" style="display:none">
      <div class="sources-label">Sources</div>
      <div id="source-pills"></div>
    </div>
  </div>
</div>

<script>
  const EXAMPLES = [
    "Who is the main subject of these books?",
    "What is Satyagraha?",
    "What is nonviolence?",
    "Key events in 1930",
    "Philosophy on truth",
  ];

  const chipsEl = document.getElementById("chips");
  EXAMPLES.forEach(e => {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = e;
    chip.onclick = () => { document.getElementById("q").value = e; ask(); };
    chipsEl.appendChild(chip);
  });

  fetch("/stats").then(r => r.json()).then(d => {
    document.getElementById("stats-label").textContent =
      `${d.total_chunks.toLocaleString()} indexed chunks ready`;
  });

  async function ask() {
    const qEl  = document.getElementById("q");
    const btn  = document.getElementById("btn");
    const card = document.getElementById("answer-card");
    const text = document.getElementById("answer-text");
    const srcs = document.getElementById("sources");
    const pills= document.getElementById("source-pills");

    const q = qEl.value.trim();
    if (!q) return;

    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span>Thinking…';
    card.classList.remove("show");

    try {
      const res  = await fetch("/ask", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ query: q }),
      });
      const data = await res.json();

      if (data.error) {
        text.innerHTML = `<span class="error">Error: ${data.error}</span>`;
      } else {
        text.textContent = data.answer;
      }

      if (data.sources && data.sources.length) {
        pills.innerHTML = data.sources
          .map(s => `<span class="source-pill">📄 ${s.source}</span>`)
          .join("");
        srcs.style.display = "block";
      } else {
        srcs.style.display = "none";
      }

      card.classList.add("show");
    } catch (err) {
      text.innerHTML = `<span class="error">Network error: ${err.message}</span>`;
      card.classList.add("show");
    } finally {
      btn.disabled = false;
      btn.textContent = "Ask";
    }
  }
</script>
</body>
</html>"""


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  BOOK Q&A SERVER")
    print("=" * 65)
    print(f"  → http://localhost:8000")
    print("=" * 65 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)