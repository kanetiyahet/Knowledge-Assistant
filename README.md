<img width="507" height="747" alt="Book Q&A Bot Screenshot" src="https://github.com/user-attachments/assets/cc5d7aa1-9cf5-42cf-9781-226276d5a124" />

# 📚 Book Q&A Assistant

**AI-Powered RAG Chatbot — Upload PDFs, ask questions, get cited answers with page numbers.**

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)](https://fastapi.tiangolo.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20DB-purple)](https://trychroma.com)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-orange)](https://ollama.ai)

---

## 📖 About

Upload any PDF books, research papers, or documents. Ask questions in natural language. Get answers with **exact book name and page citations**. 100% offline — your data never leaves your computer.

---

## ✨ Features

- 🔍 **Smart Tag Search** — Auto-adds BIRTH_INFO, SATYAGRAHA, AHIMSA tags
- 🎯 **Cross-encoder Reranking** — Best chunks ranked first
- ✍️ **Typo Correction** — "ghandhi"→"Gandhi", "pleace"→"place"
- 📚 **Source Citations** — Book Name + Page Number on every answer
- 💬 **Chat Widget** — Floating bot for any website
- 🎤 **Voice Input** — Click mic to speak (Chrome)
- 📥 **Export Chat** — Download as .txt
- 📋 **Copy Answer** — One-click copy
- 🔒 **100% Local** — No internet needed

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | HTML/CSS/JS |
| Backend | FastAPI + Python |
| AI Model | Ollama + Qwen 2.5 (3B) |
| Embeddings | all-MiniLM-L6-v2 |
| Vector DB | ChromaDB |
| Reranking | Cross-encoder (ms-marco-MiniLM) |
| Search | Hybrid (Vector + Keyword) |

---


---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Git
- [Ollama](https://ollama.ai/download)

### Setup

```bash
# 1. Clone
git clone https://github.com/kanetiyahet/Knowledge-Assistant.git
cd Knowledge-Assistant

# 2. Setup Python
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 3. Pull AI model
ollama pull qwen2.5:3b

# 4. Add books
mkdir books
# Copy your PDF files into books/ folder

# 5. Index books (choose one)
python index_data.py          # Smart chunks (Recommended)
python extract_facts.py       # Fact extraction (Alternative)

# 6. Start server
uvicorn main:app --host 0.0.0.0 --port 8000

# 7. Open chat widget
start knowledge-bot.html


Knowledge-Assistant/
├── main.py              # FastAPI backend
├── index_data.py        # PDF → Smart Chunks → ChromaDB
├── extract_facts.py     # PDF → Facts → ChromaDB
├── knowledge-bot.html   # Chat widget
├── requirements.txt     # Dependencies
├── books/               # Your PDFs
├── chroma_db/           # Chunks database
├── facts_db/            # Facts database
└── README.md
