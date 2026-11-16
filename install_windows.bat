@echo off
setlocal enabledelayedexpansion
title VoxCPM2 Lightweight - Windows Installer

echo =============================================================
echo   VoxCPM2 Lightweight - Windows 1-Click Installer
echo   Target: Burmese (my-MM) ^& English (en) Speech Synthesis
echo =============================================================

:: 1. Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.10, 3.11, or 3.12 from python.org
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('python --version') do echo [FOUND] %%i

:: 2. Check / Install FFmpeg
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo [INFO] ffmpeg not found in PATH.
    echo Trying to install via winget (Windows Package Manager)...
    winget install Gyan.FFmpeg --accept-package-agreements --accept-source-agreements >nul 2>&1
    if errorlevel 1 (
        echo [WARNING] Automatic ffmpeg install failed. Please install ffmpeg manually or via winget:
        echo   winget install Gyan.FFmpeg
    ) else (
        echo [SUCCESS] ffmpeg installed successfully!
    )
) else (
    echo [FOUND] ffmpeg is installed.
)

:: 3. Setup Virtual Environment
if not exist ".venv" (
    echo [INFO] Creating Python virtual environment in .venv...
    python -m venv .venv
)
call .venv\Scripts\activate.bat
echo [SUCCESS] Virtual environment activated.

:: 4. Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

:: 5. Detect Hardware (NVIDIA CUDA GPU vs CPU)
echo [INFO] Detecting hardware backend...
nvidia-smi >nul 2>&1
if not errorlevel 1 (
    echo [SUCCESS] NVIDIA GPU detected! Installing PyTorch with CUDA 12.1 acceleration...
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
) else (
    echo [INFO] No NVIDIA GPU detected. Installing CPU-optimized PyTorch...
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
)

echo [INFO] Installing required dependencies...
pip install -r requirements.txt

:: 6. Download Model Weights
echo [INFO] Downloading pre-pruned model weights from Hugging Face (minthura635/voxmmlite)...
python download_models.py

echo.
echo =============================================================
echo   Installation Completed Successfully!
echo   To launch the Web UI ^& Speech API:
echo   Double-click: start_webui.bat
echo =============================================================
pause
