#!/usr/bin/env python3
"""Debug GUI initialization."""
import os
import sys

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sys.path.insert(0, '/home/tinix/claude_wsl/transistordatabase')
os.chdir('/home/tinix/claude_wsl/transistordatabase')

try:
    print("Creating QApplication...")
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    print("✅ QApplication created")

    print("\nImporting MainWindow...")
    from transistordatabase.gui.gui import MainWindow
    print("✅ MainWindow imported")

    # Need to inject app into the module namespace for MainWindow to work
    import transistordatabase.gui.gui as gui_module
    gui_module.app = app

    print("\nCreating MainWindow instance...")
    window = MainWindow()
    print("✅ MainWindow instance created")

    print(f"\nWindow Title: {window.windowTitle()}")
    print(f"Menu Bar: {window.menuBar()}")
    print(f"Central Widget: {window.centralWidget()}")
    print(f"Status Bar: {window.statusBar()}")

    print("\n✅ All GUI components initialized successfully")

except Exception as e:
    import traceback
    print(f"❌ Error: {e}")
    traceback.print_exc()
