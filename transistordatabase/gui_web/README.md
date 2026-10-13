# Transistor Database - Vue.js Web Interface

This is the modern web interface for the Transistor Database, built with Vue 3 + Vite and FastAPI backend.

## Features

🌐 **Modern Web Interface** with comprehensive transistor database management:

- **🔍 Advanced Search Database**: Multi-criteria filtering, range-based property selection, manufacturer/type filters
- **📤 Professional Export Tools**: MATLAB, Simulink, PLECS, GeckoCIRCUITS, SPICE, JSON export with history
- **🧮 Topology Calculator**: Buck/Boost/Buck-Boost converter design with power loss analysis  
- **🔍 Comparison Tools**: Side-by-side transistor analysis with interactive charts
- **📊 Database Management**: Import/export, statistics, data validation
- **🌙 Dark/Light Theme**: Customizable interface with localStorage persistence
- **📱 Responsive Design**: Works on desktop, tablet, and mobile devices

## Quick Start

1. **Start Backend** with the repository's Conda environment (from the repository root):
```bash
conda activate ntbees2
python -m uvicorn transistordatabase.gui_web.backend.main:app --host 127.0.0.1 --port 8002
```
The backend launcher scripts use this full FastAPI application, including the topology routes. They do not create an environment or install dependencies.
`dev-server.bat` uses `pushd` for directory changes so the backend and frontend can also start from a UNC checkout.

2. **Start Frontend** (from root `gui_web/` directory):
```bash
npm run dev
```
Install the frontend dependencies separately before the first run.

3. **Access Application:**
- Web Interface: http://localhost:5174
- API Documentation: http://localhost:8002/docs

## Technology Stack

- **Frontend**: Vue 3, Vite, Chart.js, Axios
- **Backend**: FastAPI, Python 3.8+
- **Database**: JSON file-based with 25+ transistors
- **Styling**: CSS Variables for theming, responsive design

## Vue 3 + Vite

This template uses Vue 3 `<script setup>` SFCs. Learn more about [script setup docs](https://v3.vuejs.org/api/sfc-script-setup.html#sfc-script-setup) and [IDE Support for Vue](https://vuejs.org/guide/scaling-up/tooling.html#ide-support).

## Project Structure

```
gui_web/
├── src/
│   ├── components/          # Vue components
│   │   ├── SearchDatabase.vue      # Advanced search interface
│   │   ├── ExportingTools.vue      # Export functionality  
│   │   ├── TopologyCalculator.vue  # Converter calculations
│   │   ├── TransistorComparison.vue # Comparison tools
│   │   └── ...
│   ├── services/            # API services
│   ├── App.vue             # Main application
│   └── main.js             # Application entry
├── backend/                 # FastAPI backend package
│   └── main.py             # Full API server, including topology routes
└── package.json            # Dependencies
```
