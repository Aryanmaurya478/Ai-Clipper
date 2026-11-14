@echo off
title AI Reel Clipper
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python is not installed. Get it from https://www.python.org/downloads/
    echo During install, tick "Add Python to PATH".
    pause
    exit /b 1
)

rem Install when first run, or whenever requirements.txt has changed
fc /b requirements.txt .installed >nul 2>nul
if errorlevel 1 (
    echo Installing requirements - this can take a few minutes...
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo Install failed. Check the error above.
        pause
        exit /b 1
    )
    copy /y requirements.txt .installed >nul
)

echo.
echo Starting AI Reel Clipper at http://localhost:5000
echo Keep this window open while you use it. Close it to stop.
echo.
python app.py
pause
