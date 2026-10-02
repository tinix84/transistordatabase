# 🛠️ Web GUI Development Launch

## Development Scripts

The development launchers use the repository's `ntbees2` Conda environment for Python and run the frontend from the existing Node environment. They do not create Python environments or install dependencies.

### 📁 Location
All scripts are in: `/transistordatabase/gui_web/`

### 🚀 Main Scripts

#### 1. **`dev-server.sh`** - Main Development Script
```bash
# Start both backend and frontend (most common)
./dev-server.sh

# Start only backend (FastAPI on port 8002)
./dev-server.sh backend

# Start only frontend (Vue.js on port 5174)
./dev-server.sh frontend

# Show environment setup guidance
./dev-server.sh setup

# Show help
./dev-server.sh help
```

**Behavior:**
- Runs the full FastAPI app `transistordatabase.gui_web.backend.main:app` on port 8002, including topology routes
- Runs Vite on port 5174
- Uses `conda run -n ntbees2` for the backend
- Does not create environments or install dependencies
- The Windows launcher uses `pushd` for directory changes, including UNC checkout paths

#### 2. **`dev-server.bat`** - Windows Version
Same functionality as the shell script but for Windows users.

#### 3. **`test-env.sh`** - Legacy Environment Verification
This older helper checks a UV-based environment and is not part of the recommended Conda development launch.

#### 4. **`production.sh`** - Production Build
```bash
# Build and serve production version
./production.sh

# Build for deployment only
./production.sh static
```

#### 5. **`verify-deployment.sh`** - Vercel Readiness Check
```bash
./verify-deployment.sh
```

## 🎯 How Everything Works

### Python Environment
- The backend runs in the documented `ntbees2` Conda environment.
- Install or update Python and frontend dependencies separately from the launcher.

### Smart Directory Navigation
- Scripts auto-detect their location and navigate correctly
- Works from any directory in the project
- Handles relative paths properly

### Process Management
- Backend runs on port 8002 with FastAPI/uvicorn
- Frontend runs on port 5174 with Vite dev server
- When running both, backend starts first, then frontend
- Proper cleanup when stopping with Ctrl+C

### Error Handling
- Comprehensive checks at each step
- Clear error messages with solutions
- Graceful failure with proper cleanup

## 🚀 Quick Start Workflow

1. **First time setup:**
   ```bash
   cd transistordatabase/gui_web
   conda activate ntbees2
   # Ensure frontend dependencies are available separately.
   ```

2. **Daily development:**
   ```bash
   ./dev-server.sh        # Start both servers
   ```
   Then open:
   - Frontend: http://localhost:5174
   - Backend API: http://localhost:8002
   - API Docs: http://localhost:8002/docs

3. **Production testing:**
   ```bash
   ./production.sh        # Build and test production version
   ```

4. **Deploy to Vercel:**
   ```bash
   ./verify-deployment.sh # Check deployment readiness
   vercel                 # Deploy to Vercel
   ```

## 🔧 What The Scripts Handle Automatically

### Backend Setup:
- Runs the full `transistordatabase.gui_web.backend.main:app` FastAPI application using `ntbees2`.
- Does not install packages or create an environment.

### Frontend Setup:
- Navigates to the `gui_web` directory.
- Starts the Vite dev server with existing npm dependencies.

### Environment Management:
- Uses the documented `ntbees2` Conda environment for the backend.
- Handles cross-platform differences and directory navigation.

## 📊 Status Check

The launchers were updated to select the full FastAPI application so the Vite topology workflow reaches its API routes. Their runtime behavior was not exercised as part of that source change.
