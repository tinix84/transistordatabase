# 🚀 Development Scripts

This directory contains helper scripts for the Transistor Database web GUI. The development launchers use the repository's `ntbees2` Conda environment and do not create environments or install dependencies.

## 📋 Available Scripts

### 🔧 `dev-server.sh` (Linux/macOS)
Main development launcher:

```bash
# Start both backend and frontend
./dev-server.sh

# Start only backend
./dev-server.sh backend

# Start only frontend  
./dev-server.sh frontend

# Show environment setup guidance
./dev-server.sh install

# Show help
./dev-server.sh help
```

**Behavior:**
- Starts the full FastAPI app `transistordatabase.gui_web.backend.main:app` on port 8002, including topology routes
- Starts Vue/Vite on port 5174
- Uses `conda run -n ntbees2` for the backend
- Does not create environments or install dependencies
- The Windows launcher uses `pushd` for directory changes, including UNC checkout paths

### 🪟 `dev-server.bat` (Windows)
Windows equivalent with same functionality:

```batch
# Start both servers
dev-server.bat

# Start only backend
dev-server.bat backend

# Start only frontend
dev-server.bat frontend
```

### 🧪 `test-env.sh` (Legacy Environment Testing)
This older helper checks a UV-based environment and is not part of the recommended Conda development launch.

### 🏭 `production.sh` (Production Build)
Builds and serves production version:

```bash
# Build and serve
./production.sh

# Build only (for deployment)
./production.sh static
```

### ✅ `verify-deployment.sh` (Vercel Preparation)
Verifies Vercel deployment readiness:

```bash
./verify-deployment.sh
```

## 🎯 Quick Start

1. **First-time setup:**
   ```bash
   conda activate ntbees2
   # Install frontend dependencies separately when needed.
   ```

2. **Daily development:**
   ```bash
   ./dev-server.sh        # Start both servers
   ```

3. **Production testing:**
   ```bash
   ./production.sh        # Build and preview
   ```

4. **Deploy to Vercel:**
   ```bash
   ./verify-deployment.sh # Check readiness
   vercel                 # Deploy
   ```

## 📡 Server URLs

When running locally:
- **Frontend**: http://localhost:5174
- **Backend API**: http://localhost:8002  
- **API Documentation**: http://localhost:8002/docs

## 🔧 Requirements

For development:
- Python 3.8+
- Conda environment named `ntbees2`
- Node.js 16+
- npm

**Installation Commands:**
```bash
# Install Node.js (via package manager, if needed)
# Ubuntu/Debian:
sudo apt install nodejs npm

# macOS (via Homebrew):
brew install node

# Or download from: https://nodejs.org/
```

## 🐛 Troubleshooting

### Port Already in Use
- Frontend (5174): Stop existing Vite dev server
- Backend (8002): Stop existing FastAPI/uvicorn process

### Permission Denied
```bash
# Make scripts executable
chmod +x *.sh
```

## 🌟 Script Features

### Environment Selection
- Runs Python in the documented `ntbees2` Conda environment.
- Does not create or remove Python environments.

### Dependency Management
- Dependencies are managed separately from the launchers.
- The launcher checks for Conda, Node.js, and npm before starting.

### Process Management
- Runs backend and frontend in parallel
- Proper cleanup when stopping
- Background process handling with PID tracking

### Error Handling
- Comprehensive error checking at each step
- Helpful error messages with solutions
- Graceful failure with cleanup

### Development Workflow
- One-command launch when the Conda environment and frontend dependencies are already available
- Separate commands for backend/frontend only
- Production build testing
- Deployment preparation

## 📁 Directory Structure

```
gui_web/
├── dev-server.sh         # Main development script (Linux/macOS)
├── dev-server.bat        # Main development script (Windows)  
├── test-env.sh          # Environment verification
├── production.sh        # Production build & serve
├── verify-deployment.sh # Vercel deployment check
├── src/                 # Vue.js source code
├── backend/             # FastAPI backend package, including main.py
├── package.json         # Frontend dependencies
├── requirements.txt     # Python dependencies (for Vercel)
└── vercel.json         # Vercel deployment config
```

The launchers handle directory navigation; the Python backend runs with `conda run -n ntbees2`.
