#!/bin/bash
export QT_QPA_PLATFORM=offscreen

cd /home/tinix/claude_wsl/transistordatabase

echo "======================================"
echo "PART 1: STATIC ANALYSIS"
echo "======================================"
echo ""

echo "Step 1.1: Running Pylint on GUI code..."
echo ""
python3 -m pylint transistordatabase/gui/ --disable=C0103,C0111,R0913,R0914,R0915 2>&1 | tee pylint_results.txt
echo ""

echo "Step 1.2: Testing basic GUI import..."
python3 -c "from transistordatabase.gui.gui import MainWindow; print('✅ GUI imports successfully')"
echo ""

echo "Step 1.3: Testing mixin imports..."
python3 -c "
from transistordatabase.gui.mixins.utils_mixin import UtilsMixin
from transistordatabase.gui.mixins.settings_mixin import SettingsMixin
from transistordatabase.gui.mixins.search_mixin import SearchMixin
from transistordatabase.gui.mixins.creation_mixin import CreationMixin
from transistordatabase.gui.mixins.curve_mixin import CurveMixin
from transistordatabase.gui.mixins.export_mixin import ExportMixin
from transistordatabase.gui.mixins.comparison_mixin import ComparisonMixin
from transistordatabase.gui.mixins.topology_mixin import TopologyMixin
print('✅ All mixins import successfully')
"
echo ""

echo "======================================"
echo "PART 2: UNIT TESTING WITH PYTEST-QT"
echo "======================================"
echo ""

echo "Running all GUI tests with coverage..."
python3 -m pytest tests/test_gui_*.py -v --cov=transistordatabase/gui --cov-report=html --cov-report=term 2>&1 | tee pytest_results.txt
echo ""

echo "======================================"
echo "PART 3: GENERATING SUMMARY REPORT"
echo "======================================"
echo ""

{
    echo "=== PYQT5 GUI TEST SUMMARY REPORT ==="
    echo ""
    echo "Date: $(date)"
    echo "Platform: $(uname -s) $(uname -r)"
    echo "Python: $(python3 --version)"
    echo "PyQt5: $(python3 -c 'import PyQt5; print(PyQt5.QtCore.PYQT_VERSION_STR)')"
    echo ""
    echo "---"
    echo "STATIC ANALYSIS RESULTS"
    echo "---"
    echo ""

    if [ -f pylint_results.txt ]; then
        tail -20 pylint_results.txt
    fi
    echo ""

    echo "---"
    echo "TEST RESULTS"
    echo "---"
    echo ""

    if [ -f pytest_results.txt ]; then
        tail -30 pytest_results.txt
    fi
    echo ""

    echo "---"
    echo "SUMMARY"
    echo "---"
    echo ""
    echo "Test files created:"
    ls -1 tests/test_gui_*.py
    echo ""
    echo "Coverage report generated: htmlcov/index.html"

} | tee PYQT_GUI_TEST_REPORT.txt

echo ""
echo "======================================"
echo "TEST EXECUTION COMPLETE"
echo "======================================"
echo "Summary report saved to: PYQT_GUI_TEST_REPORT.txt"
