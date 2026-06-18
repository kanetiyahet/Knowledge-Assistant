\# 🕊️ Gandhi Knowledge Assistant



\*\*AI-Powered Digital Library \& Research Assistant for Gandhian Literature\*\*



\[!\[Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)

\[!\[FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)](https://fastapi.tiangolo.com)

\[!\[React](https://img.shields.io/badge/React-19-blue)](https://react.dev)

\[!\[Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-orange)](https://ollama.ai)



\---



\## 📖 About



A RAG-based chatbot that answers questions about Mahatma Gandhi's writings by searching through digitized books and providing answers with \*\*book name and page citations\*\*.



\---



\## ✨ Features



\- 🔍 \*\*Smart Search\*\* — Find answers across multiple books

\- 📚 \*\*Source Citations\*\* — Every answer includes book name \& page number

\- 🎯 \*\*RAG Pipeline\*\* — Retrieval-Augmented Generation for accurate answers

\- 🖥️ \*\*Beautiful UI\*\* — React-based chat interface

\- 🗄️ \*\*Vector Database\*\* — Fast semantic search with Qdrant

\- 🤖 \*\*Local LLM\*\* — Privacy-first with Ollama \& Qwen 2.5

\- 📄 \*\*PDF Processing\*\* — Automatic text extraction from books



\---



\## 🛠️ Tech Stack



| Layer | Technology |

|-------|------------|

| \*\*Frontend\*\* | React, Vite |

| \*\*Backend\*\* | FastAPI, Python |

| \*\*AI Model\*\* | Ollama, Qwen 2.5 (3B) |

| \*\*Embeddings\*\* | Sentence Transformers (all-MiniLM-L6-v2) |

| \*\*Vector DB\*\* | Qdrant |

| \*\*PDF Processing\*\* | PyMuPDF |



\---



\## 🚀 Getting Started



\### Prerequisites

\- Python 3.10+

\- Node.js 18+

\- Ollama (\[Install](https://ollama.ai))



\### Installation



```bash

\# Clone the repository

git clone https://github.com/kanetiyahet/Knowledge-Assistant.git

cd gandhi-knowledge-assistant



\# Create virtual environment

python -m venv venv

venv\\Scripts\\activate  # Windows

\# source venv/bin/activate  # Mac/Linux



\# Install Python dependencies

pip install -r requirements.txt



\# Pull the LLM model

ollama pull qwen2.5:3b



\# Install frontend dependencies

cd frontend

npm install

cd ..

