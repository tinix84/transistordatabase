"""Transistor database package file."""
__version__ = "1.0.0"
from transistordatabase.constants import *
from transistordatabase.mongodb_handling import *
from transistordatabase.checker_functions import *
from transistordatabase.helper_functions import *
from transistordatabase.data_classes import *
from transistordatabase.transistor import *
from transistordatabase.diode import *
from transistordatabase.switch import *
from transistordatabase.exceptions import *
from transistordatabase.database_manager import *
from transistordatabase.colors import *
from transistordatabase.generalplotsettings import *

# Core architecture (preferred for new code)
from transistordatabase.core.models import (  # noqa: F811
    Transistor as CoreTransistor,
    Switch as CoreSwitch,
    Diode as CoreDiode,
    TransistorMetadata,
    ElectricalRatings,
    ThermalProperties,
)
from transistordatabase.core.repository import (
    JsonTransistorRepository,
    JsonTransistorLoader,
    TransistorFactory,
)
from transistordatabase.core.adapters import (
    legacy_to_core,
    core_to_legacy_dicts,
)
