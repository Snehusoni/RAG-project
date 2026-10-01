@echo off
title DocuMind RAG — Document Intelligence Engine
echo ========================================================
echo   Launching DocuMind RAG Document Q&A Chatbot...
echo ========================================================
echo.

IF NOT EXIST ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found. Please run installation first.
    pause
    exit /b 1
)

echo Activating virtual environment and starting Streamlit...
.\.venv\Scripts\streamlit.exe run app.py
pause
