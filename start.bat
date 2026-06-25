@echo off

call venv\Scripts\activate

start "" ollama serve

timeout /t 3 >nul

python run.py

pause