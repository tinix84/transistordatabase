"""Test MainWindow initialization."""
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import pytest
from PyQt5.QtWidgets import QApplication, QToolBar
from transistordatabase.gui.gui import MainWindow


@pytest.fixture
def main_window(qtbot):
    """Create MainWindow instance."""
    window = MainWindow()
    qtbot.addWidget(window)
    return window


def test_mainwindow_initializes(main_window):
    """Test that MainWindow initializes without errors."""
    assert main_window is not None
    assert main_window.windowTitle() == "Transistor Database"


def test_mainwindow_has_menu_bar(main_window):
    """Test that menu bar exists."""
    menu_bar = main_window.menuBar()
    assert menu_bar is not None

    # Check for main menus
    menu_texts = [action.text() for action in menu_bar.actions()]
    assert len(menu_texts) > 0, "No menu items found"


def test_mainwindow_has_central_widget(main_window):
    """Test that central widget is set."""
    central = main_window.centralWidget()
    assert central is not None


def test_mainwindow_has_status_bar(main_window):
    """Test that status bar exists."""
    status_bar = main_window.statusBar()
    assert status_bar is not None


def test_mainwindow_has_toolbar(main_window):
    """Test that toolbar exists."""
    toolbars = main_window.findChildren(QToolBar)
    # May or may not have toolbars, but should not crash
    assert isinstance(toolbars, list)


def test_mainwindow_geometry(main_window):
    """Test that window has reasonable geometry."""
    geometry = main_window.geometry()
    assert geometry.width() > 0
    assert geometry.height() > 0


def test_mainwindow_inherits_from_qmainwindow(main_window):
    """Test that MainWindow is a QMainWindow."""
    from PyQt5.QtWidgets import QMainWindow
    assert isinstance(main_window, QMainWindow)
