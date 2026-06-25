@echo off

python --version >nul 2>&1
if errorlevel 1 (
    echo Install Python first:
    echo https://www.python.org/downloads/
    pause
    exit /b
)

if not exist venv (
    python -m venv venv
)

call venv\Scripts\activate

python -m pip install --upgrade pip

pip install -r requirements.txt

where ollama >nul 2>&1
if errorlevel 1 (
    echo Install Ollama first:
    echo https://ollama.com/download
    pause
    exit /b
)

start "" ollama serve
timeout /t 5 >nul

ollama pull nomic-embed-text
ollama pull qwen2.5:3b

python index.py

echo.
echo Setup Complete!
pause