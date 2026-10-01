# PowerShell Launcher for DocuMind RAG Document Q&A Chatbot
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  Launching DocuMind RAG Document Q&A Chatbot..." -ForegroundColor Yellow
Write-Host "========================================================" -ForegroundColor Cyan

$VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$StreamlitExe = Join-Path $PSScriptRoot ".venv\Scripts\streamlit.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "[ERROR] Virtual environment not found at $VenvPython" -ForegroundColor Red
    Exit 1
}

Write-Host "Starting Streamlit UI on http://localhost:8501..." -ForegroundColor Green
& $StreamlitExe run (Join-Path $PSScriptRoot "app.py")
