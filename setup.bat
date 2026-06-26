@echo off
title Gandhi Assistant Setup
color 0A

echo ==========================================
echo       GANDHI ASSISTANT SETUP
echo ==========================================
echo.

:: ------------------------------
:: Check Python
:: ------------------------------
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed.
    echo Please install Python 3.11 or newer:
    echo https://www.python.org/downloads/
    pause
    exit /b 1
)

:: ------------------------------
:: Create venv if needed
:: ------------------------------
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate

echo Updating pip...
python -m pip install --upgrade pip

echo Installing requirements...
pip install -r requirements.txt

:: ------------------------------
:: Check Ollama
:: ------------------------------
where ollama >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] Ollama is not installed.
    echo Download it from:
    echo https://ollama.com/download
    pause
    exit /b 1
)

:: ------------------------------
:: Start Ollama
:: ------------------------------
tasklist | find /i "ollama.exe" >nul
if errorlevel 1 (
    start "" ollama serve
    timeout /t 5 >nul
)

:: ------------------------------
:: Install models only if missing
:: ------------------------------
ollama list | findstr /i "nomic-embed-text" >nul
if errorlevel 1 (
    echo Downloading nomic-embed-text...
    ollama pull nomic-embed-text
)

ollama list | findstr /i "qwen2.5:3b" >nul
if errorlevel 1 (
    echo Downloading qwen2.5:3b...
    ollama pull qwen2.5:3b
)

:: ------------------------------
:: Build database only if missing
:: ------------------------------
if exist chroma_db (
    echo Existing Chroma database found.
    echo Skipping indexing.
) else (
    echo Building Chroma database...
    python index.py
)

echo.
echo ==========================================
echo Setup Complete!
echo Run start.bat to launch the application.
echo ==========================================
pause
