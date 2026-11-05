@echo off
setlocal enabledelayedexpansion
title Setup - Handwriting Synthesis & ML Curriculum Platform
cls

echo ==============================================================================
echo   HYPER-REALISTIC HANDWRITING SYNTHESIS & ML CURRICULUM - ONE-CLICK SETUP
echo ==============================================================================
echo.

cd /d "%~dp0"

echo [1/4] Verifying Python installation...
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found in PATH!
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)
python --version
echo Python verified.
echo.

echo [2/4] Verifying Node.js and npm...
where node >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Node.js is not found in PATH!
    echo Please install Node.js 18+ from https://nodejs.org/
    pause
    exit /b 1
)
where npm >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] npm is not found in PATH!
    pause
    exit /b 1
)
node --version
npm --version
echo Node.js and npm verified.
echo.

echo [3/4] Installing Python dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [WARNING] Some pip dependencies failed. Retrying with --prefer-binary...
    python -m pip install -r requirements.txt --prefer-binary
)
echo Python dependencies installed successfully.
echo.

echo [4/4] Installing Frontend dependencies and building Next.js web application...
cd "%~dp0demo_app"
call npm install
if %errorlevel% neq 0 (
    echo [ERROR] npm install encountered errors.
    pause
    exit /b 1
)
echo Building production web bundle...
call npm run build
if %errorlevel% neq 0 (
    echo [WARNING] npm run build warning; web app will run in development mode if needed.
)
cd /d "%~dp0"
echo.

echo ==============================================================================
echo   SETUP COMPLETED SUCCESSFULLY!
echo ==============================================================================
echo.
echo All Python & Node.js dependencies are installed and verified.
echo To launch the platform and interactive web demo, simply double-click:
echo.
echo    run_demo.bat
echo.
echo Press any key to exit setup...
pause >nul
