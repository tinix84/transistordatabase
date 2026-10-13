#!/bin/bash

# Transistor Database - Development Server Startup Script
# This script starts both backend and frontend servers for local development

set -e  # Exit on any error

echo "🔌 Transistor Database - Development Setup"
echo "=========================================="

# Get the script directory (gui_web folder)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "📁 Project root: $PROJECT_ROOT"
echo "📁 GUI web dir: $SCRIPT_DIR"

# These commands only print guidance and do not need runtime dependencies.
if [ "$1" = "help" ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo ""
    echo "Usage: ./dev-server.sh [command]"
    echo ""
    echo "Commands:"
    echo "  (no args)         Start both backend and frontend"
    echo "  backend|api       Start only the FastAPI backend"
    echo "  frontend|web|ui   Start only the Vue.js frontend"
    echo "  install|setup     Show environment setup guidance"
    echo "  help              Show this help message"
    echo ""
    echo "URLs:"
    echo "  Frontend:  http://localhost:5174"
    echo "  Backend:   http://localhost:8002"
    echo "  API Docs:  http://localhost:8002/docs"
    exit 0
elif [ "$1" = "install" ] || [ "$1" = "setup" ]; then
    echo ""
    echo "The launcher does not create environments or install dependencies."
    echo "Use the repository's ntbees2 Conda environment and install frontend dependencies separately."
    exit 0
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check required tools
echo ""
echo "🔍 Checking requirements..."

if ! command_exists conda; then
    echo -e "${RED}❌ Conda not found. Configure the repository's ntbees2 environment first.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Conda found${NC}"

if ! command_exists node; then
    echo -e "${RED}❌ Node.js not found. Please install Node.js 16+${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Node.js found: $(node --version)${NC}"

if ! command_exists npm; then
    echo -e "${RED}❌ npm not found. Please install npm${NC}"
    exit 1
fi
echo -e "${GREEN}✅ npm found: $(npm --version)${NC}"

# Function to start backend
start_backend() {
    echo -e "${BLUE}🚀 Starting FastAPI backend...${NC}"
    cd "$PROJECT_ROOT"
    
    echo -e "${YELLOW}📍 Backend running at: http://localhost:8002${NC}"
    echo -e "${YELLOW}📖 API docs at: http://localhost:8002/docs${NC}"
    
    conda run -n ntbees2 python -m uvicorn transistordatabase.gui_web.backend.main:app --host 127.0.0.1 --port 8002
}

# Function to start frontend
start_frontend() {
    echo -e "${BLUE}🚀 Starting Vue.js frontend...${NC}"
    cd "$SCRIPT_DIR"
    
    echo -e "${YELLOW}📍 Frontend running at: http://localhost:5174${NC}"
    
    npm run dev
}

# Check if we should run both or just one
if [ "$#" -eq 0 ]; then
    # No arguments - run both in parallel
    echo ""
    echo "🚀 Starting both backend and frontend..."
    echo -e "${YELLOW}💡 Tip: Use Ctrl+C to stop both servers${NC}"
    echo ""
    
    # Start backend in background
    start_backend &
    BACKEND_PID=$!
    
    # Wait a moment for backend to start
    sleep 3
    
    # Start frontend in foreground
    start_frontend
    
    # When frontend stops, kill backend too
    kill $BACKEND_PID 2>/dev/null || true
    
elif [ "$1" = "backend" ] || [ "$1" = "api" ]; then
    # Run only backend
    echo ""
    echo "🚀 Starting backend only..."
    start_backend
    
elif [ "$1" = "frontend" ] || [ "$1" = "web" ] || [ "$1" = "ui" ]; then
    # Run only frontend
    echo ""
    echo "🚀 Starting frontend only..."
    start_frontend
    
else
    echo -e "${RED}❌ Unknown command: $1${NC}"
    echo "Run './dev-server.sh help' for usage information"
    exit 1
fi
