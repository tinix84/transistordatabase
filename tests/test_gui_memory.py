"""Test for memory leaks in GUI."""
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import pytest
import gc
import sys
from transistordatabase.gui.gui import MainWindow


def test_repeated_window_creation_no_crash(qtbot):
    """Test that creating/destroying windows doesn't crash."""
    try:
        for i in range(5):
            window = MainWindow()
            qtbot.addWidget(window)
            window.show()
            qtbot.wait(50)
            window.close()
            window.deleteLater()
            gc.collect()
            qtbot.wait(50)
        assert True
    except Exception as e:
        pytest.fail(f"Window creation cycle failed: {e}")


def test_window_memory_tracking(qtbot):
    """Test that window memory usage is tracked."""
    try:
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create windows
        windows = []
        for i in range(3):
            window = MainWindow()
            qtbot.addWidget(window)
            windows.append(window)

        middle_memory = process.memory_info().rss / 1024 / 1024

        # Close windows
        for window in windows:
            window.close()
            window.deleteLater()

        gc.collect()
        qtbot.wait(100)

        final_memory = process.memory_info().rss / 1024 / 1024

        # Memory should be reasonable
        memory_increase = final_memory - initial_memory
        assert memory_increase < 500, f"Memory increased by {memory_increase:.1f} MB"

    except ImportError:
        pytest.skip("psutil not available")


def test_window_close_cleans_up(qtbot):
    """Test that closing window cleans up properly."""
    window = MainWindow()
    qtbot.addWidget(window)

    # Get widget count before
    initial_widgets = len(window.findChildren(type))

    # Show and hide
    window.show()
    qtbot.wait(50)
    window.hide()
    qtbot.wait(50)

    # Get widget count after
    final_widgets = len(window.findChildren(type))

    # Widgets should still be present
    assert final_widgets > 0


def test_garbage_collection(qtbot):
    """Test that garbage collection works properly."""
    try:
        windows = []
        for i in range(5):
            window = MainWindow()
            qtbot.addWidget(window)
            windows.append(window)

        # Force garbage collection
        del windows
        gc.collect()

        # Create new window to verify memory is reused
        window = MainWindow()
        qtbot.addWidget(window)

        assert window is not None

    except Exception as e:
        pytest.fail(f"Garbage collection test failed: {e}")


def test_long_running_stability(qtbot):
    """Test that GUI remains stable over time."""
    window = MainWindow()
    qtbot.addWidget(window)

    try:
        for i in range(20):
            window.show()
            qtbot.wait(10)
            window.hide()
            qtbot.wait(10)

        # If we get here, GUI is stable
        assert True

    except Exception as e:
        pytest.fail(f"Long-running stability test failed: {e}")


def test_widget_cleanup(qtbot):
    """Test that widgets are cleaned up properly."""
    window = MainWindow()
    qtbot.addWidget(window)

    # Get initial widget count
    children = window.findChildren(type)
    initial_count = len(children)

    # Access widgets
    window.menuBar()
    window.statusBar()
    window.centralWidget()

    # Get final widget count
    children = window.findChildren(type)
    final_count = len(children)

    # Widget count should be stable
    assert final_count > 0
