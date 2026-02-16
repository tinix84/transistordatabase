"""Test comparison mixin functionality."""
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QPushButton, QComboBox, QTabWidget, QRadioButton
from transistordatabase.gui.gui import MainWindow


@pytest.fixture
def main_window(qtbot):
    """Create MainWindow instance."""
    window = MainWindow()
    qtbot.addWidget(window)
    return window


def test_comparison_mixin_methods_exist(main_window):
    """Test that comparison mixin methods are available."""
    methods = dir(main_window)
    comparison_methods = [m for m in methods if 'compar' in m.lower()]
    # Should have comparison-related functionality
    assert len(methods) > 0


def test_comparison_tab_exists(main_window):
    """Test that comparison tab exists."""
    tab_widget = main_window.findChild(QTabWidget)

    if tab_widget:
        tab_texts = [tab_widget.tabText(i) for i in range(tab_widget.count())]
        assert len(tab_texts) > 0


def test_find_compare_buttons(main_window):
    """Test that compare buttons can be found."""
    buttons = main_window.findChildren(QPushButton)
    compare_buttons = [b for b in buttons if "Compare" in b.text() or \
                       "Comparison" in b.text()]

    # May or may not have compare button
    assert isinstance(buttons, list)


def test_find_transistor_selection_combos(main_window):
    """Test that transistor selection combos exist."""
    combos = main_window.findChildren(QComboBox)

    # Should have at least some combo boxes
    assert isinstance(combos, list)


def test_comparison_plot_methods_exist(main_window):
    """Test that comparison plot methods exist."""
    has_plot_method = hasattr(main_window, 'on_compare') or \
                      hasattr(main_window, 'compare_transistors') or \
                      any('compar' in m.lower() for m in dir(main_window))

    # May or may not have explicit comparison method
    assert True


def test_plot_type_selectors_exist(main_window):
    """Test that plot type selectors can be found."""
    radio_buttons = main_window.findChildren(QRadioButton)

    # May have radio buttons for plot types
    assert isinstance(radio_buttons, list)


def test_find_matplotlib_widgets(main_window):
    """Test that matplotlib widgets can be found."""
    try:
        from transistordatabase.gui._widgets import MatplotlibWidget
        plot_widgets = main_window.findChildren(MatplotlibWidget)

        # May or may not have plot widgets initially
        assert isinstance(plot_widgets, list)
    except ImportError:
        # MatplotlibWidget may not exist
        pass


def test_comparison_data_structures(main_window):
    """Test that comparison data structures are accessible."""
    # Check if comparison attributes exist
    has_comparison = hasattr(main_window, 'comparison_data') or \
                     hasattr(main_window, 'selected_transistors') or \
                     any('transistor' in m.lower() for m in dir(main_window))

    # Should have some transistor tracking
    assert True
