@echo off
REM Transistor Database - Development Server for Windows
REM This script starts both backend and frontend servers

setlocal enabledelayedexpansion

echo 🔌 Transistor Database - Development Setup (Windows)
echo =====================================================

REM Get script directory
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%..\.."

echo 📁 Project root: %PROJECT_ROOT%
echo 📁 GUI web dir: %SCRIPT_DIR%

REM Check required tools
echo.
echo 🔍 Checking requirements...

where conda >nul 2>&1
if errorlevel 1 (
    echo ❌ Conda not found. Configure the repository's ntbees2 environment first.
    pause
    exit /b 1
)
echo ✅ Conda found

where node >nul 2>&1
if errorlevel 1 (
    echo ❌ Node.js not found. Please install Node.js 16+
    pause
    exit /b 1
)
echo ✅ Node.js found

where npm >nul 2>&1
if errorlevel 1 (
    echo ❌ npm not found. Please install npm
    pause
    exit /b 1
)
echo ✅ npm found

REM Check arguments
if "%1"=="backend" goto start_backend
if "%1"=="api" goto start_backend
if "%1"=="frontend" goto start_frontend
if "%1"=="web" goto start_frontend
if "%1"=="ui" goto start_frontend
if "%1"=="install" goto install_only
if "%1"=="setup" goto install_only
if "%1"=="help" goto show_help
if "%1"=="-h" goto show_help
if "%1"=="--help" goto show_help
if not "%1"=="" goto unknown_command

REM Default: start both
echo.
echo 🚀 Starting both backend and frontend...
echo 💡 Tip: Use Ctrl+C to stop servers
echo.

REM pushd maps UNC paths to a temporary drive in the child cmd context.
start "Backend Server" cmd /k "pushd ""%PROJECT_ROOT%"" && echo 🚀 Starting FastAPI backend... && echo 📍 Backend: http://localhost:8002 && echo 📖 API docs: http://localhost:8002/docs && conda run -n ntbees2 python -m uvicorn transistordatabase.gui_web.backend.main:app --host 127.0.0.1 --port 8002"

REM Wait for backend to start
timeout /t 3 /nobreak >nul

REM Start frontend
pushd "%SCRIPT_DIR%"
if errorlevel 1 (
    echo ❌ Could not enter the GUI directory.
    exit /b 1
)
echo 🚀 Starting Vue.js frontend...
echo 📍 Frontend: http://localhost:5174
npm run dev
set "FRONTEND_STATUS=%ERRORLEVEL%"
popd
goto end

:start_backend
echo.
echo 🚀 Starting backend only...
echo 📍 Backend running at: http://localhost:8002
echo 📖 API docs at: http://localhost:8002/docs
pushd "%PROJECT_ROOT%"
if errorlevel 1 (
    echo ❌ Could not enter the repository directory.
    exit /b 1
)
conda run -n ntbees2 python -m uvicorn transistordatabase.gui_web.backend.main:app --host 127.0.0.1 --port 8002
set "BACKEND_STATUS=%ERRORLEVEL%"
popd
goto end

:start_frontend
echo.
echo 🚀 Starting frontend only...
pushd "%SCRIPT_DIR%"
if errorlevel 1 (
    echo ❌ Could not enter the GUI directory.
    exit /b 1
)
echo 📍 Frontend running at: http://localhost:5174
npm run dev
set "FRONTEND_STATUS=%ERRORLEVEL%"
popd
goto end

:install_only
echo.
echo The launcher does not create environments or install dependencies.
echo Use the repository's ntbees2 Conda environment and install frontend dependencies separately.
goto end

:show_help
echo.
echo Usage: dev-server.bat [command]
echo.
echo Commands:
echo   (no args)         Start both backend and frontend
echo   backend/api       Start only the FastAPI backend
echo   frontend/web/ui   Start only the Vue.js frontend
echo   install/setup     Show environment setup guidance
echo   help              Show this help message
echo.
echo URLs:
echo   Frontend:  http://localhost:5174
echo   Backend:   http://localhost:8002
echo   API Docs:  http://localhost:8002/docs
goto end

:unknown_command
echo ❌ Unknown command: %1
echo Run 'dev-server.bat help' for usage information
goto end

:end
pause
