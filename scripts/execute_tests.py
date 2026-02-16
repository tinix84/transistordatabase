#!/usr/bin/env python3
"""Direct pytest execution without bash."""
import os
import sys

# Set environment for headless testing
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# Add repo to path
sys.path.insert(0, '/home/tinix/claude_wsl/transistordatabase')
os.chdir('/home/tinix/claude_wsl/transistordatabase')

import pytest
from datetime import datetime

def main():
    """Run tests directly using pytest API."""

    print("\n" + "="*70)
    print("PYQT5 GUI TEST EXECUTION")
    print("="*70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"QT Platform: {os.environ.get('QT_QPA_PLATFORM', 'default')}")
    print("")

    # Part 1: Static Analysis - Import Tests
    print("="*70)
    print("PART 1: STATIC ANALYSIS & IMPORTS")
    print("="*70)

    try:
        print("\n1.1 Testing GUI import...")
        from transistordatabase.gui.gui import MainWindow
        print("✅ GUI imports successfully")
    except Exception as e:
        print(f"❌ GUI import failed: {e}")
        return 1

    try:
        print("\n1.2 Testing mixin imports...")
        from transistordatabase.gui.mixins.utils_mixin import UtilitiesMixin
        from transistordatabase.gui.mixins.settings_mixin import SettingsMixin
        from transistordatabase.gui.mixins.search_mixin import SearchDatabaseMixin
        from transistordatabase.gui.mixins.creation_mixin import TransistorCreationMixin
        from transistordatabase.gui.mixins.curve_mixin import CurveManagementMixin
        from transistordatabase.gui.mixins.export_mixin import ExportToolsMixin
        from transistordatabase.gui.mixins.comparison_mixin import ComparisonToolsMixin
        from transistordatabase.gui.mixins.topology_mixin import TopologyCalculatorMixin
        print("✅ All 8 mixins import successfully")
    except Exception as e:
        print(f"❌ Mixin import failed: {e}")
        return 1

    try:
        print("\n1.3 Testing widgets import...")
        from transistordatabase.gui._widgets import MatplotlibWidget
        print("✅ Widgets import successfully")
    except Exception as e:
        print(f"❌ Widgets import failed: {e}")
        return 1

    # Part 2: Unit Testing
    print("\n" + "="*70)
    print("PART 2: UNIT TESTING WITH PYTEST")
    print("="*70)
    print("")

    # Run pytest on all GUI test files
    test_files = [
        'tests/test_gui_imports.py',
        'tests/test_gui_mainwindow.py',
        'tests/test_gui_search_mixin.py',
        'tests/test_gui_curve_mixin.py',
        'tests/test_gui_comparison_mixin.py',
        'tests/test_gui_export_mixin.py',
        'tests/test_gui_topology_mixin.py',
        'tests/test_gui_performance.py',
        'tests/test_gui_memory.py',
    ]

    # Run tests with pytest
    argv = [
        'tests/test_gui_imports.py',
        'tests/test_gui_mainwindow.py',
        'tests/test_gui_search_mixin.py',
        'tests/test_gui_curve_mixin.py',
        'tests/test_gui_comparison_mixin.py',
        'tests/test_gui_export_mixin.py',
        'tests/test_gui_topology_mixin.py',
        'tests/test_gui_performance.py',
        'tests/test_gui_memory.py',
        '-v',
        '--tb=short',
        '--cov=transistordatabase/gui',
        '--cov-report=term-missing',
    ]

    print(f"Running {len(test_files)} test modules...")
    print("")

    exit_code = pytest.main(argv)

    print("\n" + "="*70)
    print("TEST EXECUTION COMPLETE")
    print("="*70)
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Exit Code: {exit_code}")

    return exit_code

if __name__ == '__main__':
    sys.exit(main())
