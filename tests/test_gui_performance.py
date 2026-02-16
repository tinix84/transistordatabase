"""Test GUI performance and responsiveness."""
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import pytest
import time
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLineEdit, QPushButton
from transistordatabase.gui.gui import MainWindow


@pytest.fixture
def main_window(qtbot):
    """Create MainWindow instance."""
    window = MainWindow()
    qtbot.addWidget(window)
    return window


def test_window_launch_time(qtbot):
    """Test that window launches in reasonable time."""
    start = time.time()

    window = MainWindow()
    qtbot.addWidget(window)

    launch_time = time.time() - start

    # Window should launch in reasonable time (less than 5 seconds)
    assert launch_time < 5.0, f"Window took {launch_time:.2f}s to launch"


def test_search_responsiveness(qtbot, main_window):
    """Test that searching doesn't cause hangs."""
    search_input = main_window.findChild(QLineEdit, "search_input")

    if search_input:
        start = time.time()

        # Type search query
        qtbot.keyClicks(search_input, "TEST")

        # Process events
        from PyQt5.QtWidgets import QApplication
        app = QApplication.instance()
        if app:
            app.processEvents()

        search_time = time.time() - start

        # Search should complete quickly
        assert search_time < 2.0, f"Search took {search_time:.2f}s"


def test_window_shows_without_crash(qtbot, main_window):
    """Test that window shows and doesn't crash."""
    try:
        main_window.show()
        qtbot.wait(100)
        main_window.hide()
        assert True
    except Exception as e:
        pytest.fail(f"Window show caused error: {e}")


def test_multiple_window_launches(qtbot):
    """Test that multiple windows can be created."""
    windows = []
    for i in range(3):
        window = MainWindow()
        qtbot.addWidget(window)
        windows.append(window)

    assert len(windows) == 3

    for window in windows:
        window.close()


def test_widget_access_performance(qtbot, main_window):
    """Test that accessing widgets is fast."""
    start = time.time()

    # Access various child widgets
    for _ in range(10):
        main_window.findChildren(QLineEdit)
        main_window.findChildren(QPushButton)
        main_window.menuBar()
        main_window.statusBar()
        main_window.centralWidget()

    access_time = time.time() - start

    # Widget access should be very fast
    assert access_time < 1.0, f"Widget access took {access_time:.2f}s"


def test_window_geometry_update(qtbot, main_window):
    """Test that window geometry updates without crash."""
    start = time.time()

    # Update geometry
    main_window.setGeometry(0, 0, 1024, 768)
    main_window.setGeometry(100, 100, 800, 600)
    main_window.setGeometry(50, 50, 1280, 1024)

    update_time = time.time() - start

    assert update_time < 1.0


def test_window_close_performance(qtbot):
    """Test that window closes quickly."""
    window = MainWindow()
    qtbot.addWidget(window)

    start = time.time()

    window.close()

    close_time = time.time() - start

    assert close_time < 1.0, f"Window close took {close_time:.2f}s"
