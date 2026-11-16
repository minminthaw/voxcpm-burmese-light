@echo off
setlocal
title VoxCPM2 Burmese Lightweight - Web UI & Speech API

cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo Please run install_windows.bat first to install dependencies and models.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

set PORT=7860
if not "%~1"=="" set PORT=%~1

echo =============================================================
echo Starting VoxCPM2 Lightweight Studio ^& Speech API Server...
echo   - Web UI:    http://localhost:%PORT%
echo   - TTS API:   POST http://localhost:%PORT%/v1/audio/speech
echo   - Memory:    ~3.5 GB (INT8 Dynamic Mode Active)
echo =============================================================

start "" "http://localhost:%PORT%"

python webui.py --port %PORT%
if errorlevel 1 (
    echo.
    echo [ERROR] Server exited with an error.
    pause
)
