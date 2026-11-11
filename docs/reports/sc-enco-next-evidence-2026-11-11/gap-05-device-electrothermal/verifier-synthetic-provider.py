"""Test-only in-memory data provider; all calculations use the actual API core."""
from types import SimpleNamespace
from pathlib import Path
import sys

FEATURE_ROOT = Path('/home/tinix/claude_wsl/transistordatabase-sc-enco-next-20261111')
sys.path.insert(0, str(FEATURE_ROOT))

import numpy as np
from fastapi import Request
import uvicorn

from transistordatabase.core.models import (
    ChannelCharacteristics, ElectricalRatings, MeasurementEvidence, SwitchingLossData,
    ThermalProperties, Transistor, TransistorMetadata,
)
from transistordatabase.gui_web.backend import main
from transistordatabase.gui_web.backend.device_electrothermal_routes import device_repository

assert Path(main.__file__).resolve().is_relative_to(FEATURE_ROOT)

SYNTHETIC_ID = 'Verifier_Synthetic_Electrothermal'


def synthetic_device():
    """Return fresh analytic curves; no measurements or repository files are used."""
    device = Transistor(
        metadata=TransistorMetadata(name=SYNTHETIC_ID, type='MOSFET', author='Verifier',
                                    manufacturer='Synthetic fixture', housing_type='Analytic'),
        electrical=ElectricalRatings(1200, 100, 50, 175),
        thermal=ThermalProperties(.001, .001, .1, .1, .1))
    # R(T) = 1 + .01*(T-25) ohm. At 2 A, D=1, Ts=25 C and Rtheta=2 K/W:
    # deltaT = 8/(1-.08) K. This is an analytic verification fixture.
    device.switch.channel_data = [
        ChannelCharacteristics(t, np.array([[0, 4 * (1 + .01 * (t - 25))], [0, 4]]), 15)
        for t in (25, 125)]
    for attribute, event, gate, energy in [('e_on_data', 'turn_on', 15, .002),
                                           ('e_off_data', 'turn_off', -15, .003)]:
        records = []
        for lower_current in (0, 1):
            for temperature in (25, 125):
                records.append(SwitchingLossData(
                    'graph_i_e', temperature, 400, gate, r_g=3.3, v_g_off=None,
                    graph_i_e=np.array([[lower_current, 4], [energy, energy]]),
                    comment='Synthetic analytic fixture; no measured-device accuracy claim.',
                    commutation_device='VerifierRectifier', load_inductance=1e-3,
                    commutation_inductance=1e-8, measurement_testbench='AnalyticFixtureRig',
                    evidence=MeasurementEvidence(method='analytic-fixture', event=event,
                                                  convention='synthetic passive heat')))
        setattr(device.switch, attribute, records)
    return device


async def acceptance_repository(request: Request):
    """Override only the new provider for one explicit synthetic identity."""
    name = request.path_params.get('transistor_id')
    if name is None and request.method == 'POST':
        try:
            body = await request.json()
            name = body.get('transistor_id') if isinstance(body, dict) else None
        except ValueError:
            name = None
    if name == SYNTHETIC_ID:
        return SimpleNamespace(get_by_name=lambda value: synthetic_device() if value == SYNTHETIC_ID else None)
    # Actual device requests still use the application's existing file repository.
    return main._repo


main.app.dependency_overrides[device_repository] = acceptance_repository

if __name__ == '__main__':
    uvicorn.run(main.app, host='127.0.0.1', port=8004, reload=False)
