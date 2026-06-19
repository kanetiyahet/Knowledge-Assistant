---

## 🚀 Quick Start (5 Minutes)

### Prerequisites
- ✅ Python 3.10+ installed
- ✅ Git installed
- ✅ 8GB+ RAM (for LLM)

### Setup Commands
```bash
# 1. Clone
git clone https://github.com/kanetiyahet/Knowledge-Assistant.git
cd Knowledge-Assistant

# 2. Setup Python (Windows)
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 3. Install Ollama & Pull Model
# Download Ollama from https://ollama.ai
ollama pull qwen2.5:3b

# 4. Add books to folder
mkdir books
# Copy PDFs into books/ folder

# 5. Index books
python index_data.py

# 6. Start backend
uvicorn main:app --host 0.0.0.0 --port 8000

# 7. Open chatbot widget
start knowledge-bot.html