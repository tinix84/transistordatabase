"""Test GUI imports for circular dependencies."""
import pytest


def test_gui_imports():
    """Test that GUI module imports without circular dependencies."""
    try:
        from transistordatabase.gui.gui import MainWindow
        from transistordatabase.gui.mixins import curve_mixin
        from transistordatabase.gui.mixins import comparison_mixin
        from transistordatabase.gui.mixins import topology_mixin
        assert MainWindow is not None
        assert curve_mixin is not None
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


def test_mixin_independence():
    """Test that mixins can be imported independently."""
    from transistordatabase.gui.mixins.utils_mixin import UtilitiesMixin
    from transistordatabase.gui.mixins.settings_mixin import SettingsMixin
    from transistordatabase.gui.mixins.search_mixin import SearchDatabaseMixin
    from transistordatabase.gui.mixins.creation_mixin import TransistorCreationMixin
    from transistordatabase.gui.mixins.curve_mixin import CurveManagementMixin
    from transistordatabase.gui.mixins.export_mixin import ExportToolsMixin
    from transistordatabase.gui.mixins.comparison_mixin import ComparisonToolsMixin
    from transistordatabase.gui.mixins.topology_mixin import TopologyCalculatorMixin

    assert UtilitiesMixin is not None
    assert SettingsMixin is not None
    assert SearchDatabaseMixin is not None
    assert TransistorCreationMixin is not None
    assert CurveManagementMixin is not None
    assert ExportToolsMixin is not None
    assert ComparisonToolsMixin is not None
    assert TopologyCalculatorMixin is not None


def test_widgets_import():
    """Test that widgets module imports."""
    from transistordatabase.gui._widgets import MatplotlibWidget
    from transistordatabase.gui._widgets import PopOutPlotWindow

    assert MatplotlibWidget is not None
    assert PopOutPlotWindow is not None


def test_api_client_import():
    """Test that API client imports."""
    from transistordatabase.gui.api_client import TransistorApiClient

    assert TransistorApiClient is not None
