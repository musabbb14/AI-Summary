@echo off
title AI Document Summarizer
cd /d "%~dp0"
echo.
echo  Starting AI Document Summarization System...
echo  The app will open at: http://127.0.0.1:7860
echo.
echo  Auto-reload is ON — any code changes apply instantly.
echo  Keep this window open while using the app.
echo  Close this window to stop the app.
echo.
call venv\Scripts\activate.bat
python app.py
pause
