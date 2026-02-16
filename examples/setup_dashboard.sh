#!/bin/bash
# Setup script for Transistor Database Performance Dashboard
# Run this before launching the Jupyter notebook

set -e  # Exit on error

echo "=========================================="
echo "Transistor Database Dashboard Setup"
echo "=========================================="
echo ""

# Check Python version
echo "[1/5] Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "  Found Python $python_version"
if [[ "$python_version" < "3.10" ]]; then
    echo "  ⚠ Warning: Python 3.10+ recommended"
fi
echo ""

# Install core dependencies
echo "[2/5] Installing core dependencies..."
pip install --quiet --upgrade pip
pip install --quiet numpy pandas matplotlib scipy
echo "  ✓ NumPy, Pandas, Matplotlib, SciPy installed"
echo ""

# Install Plotly (for interactive plots)
echo "[3/5] Installing Plotly..."
pip install --quiet plotly
echo "  ✓ Plotly installed"
echo ""

# Install Jupyter and widgets
echo "[4/5] Installing Jupyter and ipywidgets..."
pip install --quiet jupyter jupyterlab ipywidgets
jupyter nbextension enable --py widgetsnbextension --sys-prefix 2>/dev/null || true
echo "  ✓ Jupyter and widgets installed"
echo ""

# Verify transistordatabase
echo "[5/5] Verifying transistordatabase..."
if python3 -c "from transistordatabase.core import Transistor" 2>/dev/null; then
    echo "  ✓ transistordatabase accessible"
else
    echo "  ⚠ transistordatabase not found. Installing..."
    pip install -e .
    echo "  ✓ transistordatabase installed"
fi
echo ""

# Check database directory
db_dir="transistors_merged"
if [ -d "$db_dir" ]; then
    num_files=$(ls -1 "$db_dir"/*.json 2>/dev/null | wc -l)
    echo "📊 Database: $num_files transistors found in $db_dir/"
else
    echo "⚠ Database directory not found: $db_dir/"
    echo "  Please ensure transistor JSON files are in this directory"
fi
echo ""

echo "=========================================="
echo "✓ Setup complete!"
echo "=========================================="
echo ""
echo "To launch the dashboard:"
echo "  jupyter notebook transistordatabase_performance_dashboard.ipynb"
echo ""
echo "Or with JupyterLab:"
echo "  jupyter lab transistordatabase_performance_dashboard.ipynb"
echo ""
echo "For more information, see DASHBOARD_README.md"
echo ""
