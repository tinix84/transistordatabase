"""
Switching-pair data models for the unified transistor database.

Every entry represents a switching pair — the minimum unit of a converter leg.
A single device is the special case where high_side.device_id == low_side.device_id.

Switching loss is a commutation-system phenomenon, not a datasheet scalar.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ConductionCurve:
    """V-I conduction curve at a specific temperature."""

    t_j: float  # Junction temperature in °C
    on_state_voltage: list[float] = field(default_factory=list)  # Voltage points (V)
    on_state_current: list[float] = field(default_factory=list)  # Current points (A)


@dataclass
class ConductionData:
    """Temperature-dependent conduction characteristics for one device."""

    temperatures: list[float] = field(default_factory=list)  # °C
    curves: list[ConductionCurve] = field(default_factory=list)


@dataclass
class ThermalElement:
    """Single RC element in Foster/Cauer thermal model."""

    r: float  # Thermal resistance (K/W)
    c: float  # Thermal capacitance (J/K)


@dataclass
class ThermalModel:
    """Foster or Cauer thermal RC network."""

    type: str = "foster"  # "foster" or "cauer"
    elements: list[ThermalElement] = field(default_factory=list)


@dataclass
class DeviceThermalProperties:
    """Thermal properties for one device in a switching pair."""

    r_th_jc: float = 0.0  # Junction-case thermal resistance (K/W)
    r_th_cs: float = 0.0  # Case-sink thermal resistance (K/W)
    r_th_common: float = 0.0  # Common thermal resistance (K/W)
    housing_area: float = 0.0  # Housing area (m²)
    cooling_area: float = 0.0  # Cooling area (m²)
    interface_area_forward: float = 0.0  # Forward thermal interface area (m²)
    interface_area_reverse: float = 0.0  # Reverse thermal interface area (m²)
    thermal_model: ThermalModel | None = None


@dataclass
class DeviceMetadata:
    """Metadata for one device in a switching pair."""

    name: str = ""
    type: str = ""  # MOSFET, IGBT, SiC-MOSFET, GaN, Diode
    manufacturer: str = ""
    housing_type: str = ""
    datasheet_hyperlink: str = ""
    datasheet_date: str = ""
    datasheet_version: str = ""
    cost: float | None = None  # USD
    weight: float | None = None  # kg


@dataclass
class DeviceElectricalRatings:
    """Electrical ratings for one device."""

    v_abs_max: float = 0.0  # Maximum blocking voltage (V)
    i_abs_max: float = 0.0  # Maximum current (A)
    i_cont: float = 0.0  # Continuous current (A)
    t_j_max: float = 175.0  # Maximum junction temperature (°C)


@dataclass
class CapacitanceData:
    """Voltage-dependent capacitance curves."""

    c_oss: list[dict[str, Any]] = field(default_factory=list)
    c_iss: list[dict[str, Any]] = field(default_factory=list)
    c_rss: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PairSide:
    """One side (high or low) of a switching pair."""

    device_id: str = ""
    count: int = 1  # Number of parallel devices
    role: str = ""  # "switch", "body_diode", "diode", "synchronous_switch"

    metadata: DeviceMetadata = field(default_factory=DeviceMetadata)
    electrical_ratings: DeviceElectricalRatings = field(default_factory=DeviceElectricalRatings)
    thermal_properties: DeviceThermalProperties = field(default_factory=DeviceThermalProperties)
    conduction: ConductionData = field(default_factory=ConductionData)
    capacitance: CapacitanceData = field(default_factory=CapacitanceData)
    gate_charge: list[dict[str, Any]] = field(default_factory=list)
    soa: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SwitchingTestConditions:
    """Test conditions for switching loss measurement."""

    gate_resistance_on: float = 0.0  # Ω
    gate_resistance_off: float = 0.0  # Ω
    gate_voltage_on: float = 15.0  # V
    gate_voltage_off: float = -5.0  # V
    dc_bus_voltage: float = 0.0  # V
    dead_time: float = 0.0  # s
    parasitic_inductance: float = 0.0  # H


@dataclass
class SwitchingEnergyData:
    """Switching energy curves (E vs I) at multiple temperatures."""

    temperatures: list[float] = field(default_factory=list)  # °C
    current_axis: list[float] = field(default_factory=list)  # A
    voltage_axis: list[float] = field(default_factory=list)  # V
    energy: dict[str, list[float]] = field(default_factory=dict)
    # energy keys are temperature strings, e.g. {"25.0": [e1, e2, ...], "150.0": [...]}


@dataclass
class SwitchingData:
    """Pair-level switching loss data — the core of the switching-pair concept."""

    source: str = "datasheet"  # "datasheet", "dpt_measurement", "ltspice_simulation", "plecs_model"
    source_details: dict[str, Any] = field(default_factory=dict)
    test_conditions: SwitchingTestConditions = field(default_factory=SwitchingTestConditions)
    turn_on: SwitchingEnergyData = field(default_factory=SwitchingEnergyData)
    turn_off: SwitchingEnergyData = field(default_factory=SwitchingEnergyData)
    reverse_recovery: SwitchingEnergyData = field(default_factory=SwitchingEnergyData)


@dataclass
class PLECSModel:
    """PLECS analytical model formulas (if available from XML)."""

    turn_on_formula: str | None = None
    turn_off_formula: str | None = None
    conduction_formula: str | None = None


@dataclass
class MigrationMetadata:
    """Tracks data provenance for each switching pair."""

    source_files: list[str] = field(default_factory=list)
    migration_date: str = ""
    schema_version: str = "1.0"
    quality_score: float = 0.0


@dataclass
class SwitchingPair:
    """
    A switching pair — the fundamental unit of the transistor database.

    Represents two devices forming a converter leg (half-bridge).
    A single device is the special case where high_side.device_id == low_side.device_id.
    """

    pair_id: str = ""
    pair_type: str = ""  # mosfet_self, igbt_self, mosfet_plus_diode, mosfet_plus_mosfet, parallel_self, etc.
    topology: str = "half_bridge"

    high_side: PairSide = field(default_factory=PairSide)
    low_side: PairSide = field(default_factory=PairSide)
    switching_data: SwitchingData = field(default_factory=SwitchingData)

    traces: dict[str, Any] = field(default_factory=dict)  # Optional curve trace data
    plecs_model: PLECSModel = field(default_factory=PLECSModel)
    migration: MigrationMetadata = field(default_factory=MigrationMetadata)

    @property
    def is_self_pair(self) -> bool:
        """Check if this is a self-pair (single device with itself)."""
        return self.high_side.device_id == self.low_side.device_id

    @property
    def is_parallel(self) -> bool:
        """Check if this involves parallel devices."""
        return self.high_side.count > 1 or self.low_side.count > 1

    @property
    def devices(self) -> list[str]:
        """Get unique device IDs in this pair."""
        ids = [self.high_side.device_id]
        if self.low_side.device_id != self.high_side.device_id:
            ids.append(self.low_side.device_id)
        return ids

    def validate(self) -> dict[str, Any]:
        """Validate this switching pair and return errors/warnings."""
        errors = []
        warnings = []

        # Required fields
        if not self.pair_id:
            errors.append("Missing pair_id")
        if not self.pair_type:
            errors.append("Missing pair_type")
        if not self.high_side.device_id:
            errors.append("Missing high_side.device_id")
        if not self.low_side.device_id:
            errors.append("Missing low_side.device_id")

        # Valid pair types
        valid_types = {
            'mosfet_self', 'igbt_self', 'sic-mosfet_self', 'gan_self', 'diode_self',
            'mosfet_plus_diode', 'igbt_plus_diode', 'mosfet_plus_mosfet',
            'parallel_self', 'parallel_combo',
        }
        if self.pair_type and self.pair_type not in valid_types:
            warnings.append(f"Unknown pair_type: {self.pair_type}")

        # Electrical ratings sanity checks
        for side_name, side in [('high_side', self.high_side), ('low_side', self.low_side)]:
            v = side.electrical_ratings.v_abs_max
            if v > 0 and (v < 10 or v > 20000):
                warnings.append(f"{side_name}.v_abs_max={v}V outside typical range [10, 20000]")
            i = side.electrical_ratings.i_abs_max
            if i > 0 and (i < 0.01 or i > 50000):
                warnings.append(f"{side_name}.i_abs_max={i}A outside typical range [0.01, 50000]")

        # Switching data array consistency
        for phase_name in ['turn_on', 'turn_off', 'reverse_recovery']:
            phase = getattr(self.switching_data, phase_name)
            n_current = len(phase.current_axis)
            if n_current > 0:
                for temp_key, energies in phase.energy.items():
                    if len(energies) != n_current:
                        errors.append(
                            f"switching_data.{phase_name}.energy[{temp_key}] has {len(energies)} "
                            f"values but current_axis has {n_current}"
                        )

        # Conduction data consistency
        for side_name, side in [('high_side', self.high_side), ('low_side', self.low_side)]:
            for curve in side.conduction.curves:
                if len(curve.on_state_voltage) != len(curve.on_state_current):
                    errors.append(
                        f"{side_name}.conduction curve at {curve.t_j}°C: "
                        f"voltage ({len(curve.on_state_voltage)}) and current ({len(curve.on_state_current)}) "
                        f"array lengths don't match"
                    )

        score = self._calculate_quality_score()

        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'quality_score': score,
        }

    def _calculate_quality_score(self) -> float:
        """Calculate data completeness score (0-100)."""
        points = 0
        max_points = 0

        # Metadata (10 pts each)
        for side in [self.high_side, self.low_side]:
            max_points += 30
            if side.metadata.name:
                points += 10
            if side.metadata.type:
                points += 10
            if side.metadata.manufacturer:
                points += 10

        # Electrical ratings (10 pts each)
        max_points += 20
        if self.high_side.electrical_ratings.v_abs_max > 0:
            points += 10
        if self.high_side.electrical_ratings.i_abs_max > 0:
            points += 10

        # Conduction data (20 pts)
        max_points += 20
        if self.high_side.conduction.curves:
            points += 20

        # Switching data (30 pts)
        max_points += 30
        if self.switching_data.turn_on.current_axis:
            points += 15
        if self.switching_data.turn_off.current_axis:
            points += 15

        # Thermal (10 pts)
        max_points += 10
        if self.high_side.thermal_properties.r_th_jc > 0:
            points += 10

        return (points / max_points * 100) if max_points > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SwitchingPair:
        """Deserialize from dictionary."""
        pair = cls()
        pair.pair_id = data.get('pair_id', '')
        pair.pair_type = data.get('pair_type', '')
        pair.topology = data.get('topology', 'half_bridge')

        # High side
        hs = data.get('high_side', {})
        pair.high_side = _parse_pair_side(hs)

        # Low side
        ls = data.get('low_side', {})
        pair.low_side = _parse_pair_side(ls)

        # Switching data
        sw = data.get('switching_data', {})
        pair.switching_data = _parse_switching_data(sw)

        # Optional fields
        pair.traces = data.get('traces', {})
        plecs = data.get('plecs_model', {})
        pair.plecs_model = PLECSModel(
            turn_on_formula=plecs.get('turn_on_formula'),
            turn_off_formula=plecs.get('turn_off_formula'),
            conduction_formula=plecs.get('conduction_formula'),
        )

        mig = data.get('migration', {})
        pair.migration = MigrationMetadata(
            source_files=mig.get('source_files', []),
            migration_date=mig.get('migration_date', ''),
            schema_version=mig.get('schema_version', '1.0'),
            quality_score=mig.get('quality_score', 0.0),
        )

        return pair


def _parse_pair_side(data: dict[str, Any]) -> PairSide:
    """Parse a PairSide from dict."""
    side = PairSide()
    side.device_id = data.get('device_id', '')
    side.count = data.get('count', 1)
    side.role = data.get('role', '')

    meta = data.get('metadata', {})
    side.metadata = DeviceMetadata(
        name=meta.get('name', ''),
        type=meta.get('type', ''),
        manufacturer=meta.get('manufacturer', ''),
        housing_type=meta.get('housing_type', ''),
        datasheet_hyperlink=meta.get('datasheet_hyperlink', ''),
        datasheet_date=meta.get('datasheet_date', ''),
        datasheet_version=meta.get('datasheet_version', ''),
        cost=meta.get('cost'),
        weight=meta.get('weight'),
    )

    elec = data.get('electrical_ratings', {})
    side.electrical_ratings = DeviceElectricalRatings(
        v_abs_max=elec.get('v_abs_max', 0.0),
        i_abs_max=elec.get('i_abs_max', 0.0),
        i_cont=elec.get('i_cont', 0.0),
        t_j_max=elec.get('t_j_max', 175.0),
    )

    therm = data.get('thermal_properties', {})
    tm = therm.get('thermal_model')
    thermal_model = None
    if tm:
        thermal_model = ThermalModel(
            type=tm.get('type', 'foster'),
            elements=[ThermalElement(r=e['r'], c=e['c']) for e in tm.get('elements', [])],
        )
    side.thermal_properties = DeviceThermalProperties(
        r_th_jc=therm.get('r_th_jc', 0.0),
        r_th_cs=therm.get('r_th_cs', 0.0),
        r_th_common=therm.get('r_th_common', 0.0),
        housing_area=therm.get('housing_area', 0.0),
        cooling_area=therm.get('cooling_area', 0.0),
        interface_area_forward=therm.get('interface_area_forward', 0.0),
        interface_area_reverse=therm.get('interface_area_reverse', 0.0),
        thermal_model=thermal_model,
    )

    cond = data.get('conduction', {})
    side.conduction = ConductionData(
        temperatures=cond.get('temperatures', []),
        curves=[
            ConductionCurve(
                t_j=c.get('t_j', 25.0),
                on_state_voltage=c.get('on_state_voltage', []),
                on_state_current=c.get('on_state_current', []),
            )
            for c in cond.get('curves', [])
        ],
    )

    cap = data.get('capacitance', {})
    side.capacitance = CapacitanceData(
        c_oss=cap.get('c_oss', []),
        c_iss=cap.get('c_iss', []),
        c_rss=cap.get('c_rss', []),
    )

    side.gate_charge = data.get('gate_charge', [])
    side.soa = data.get('soa', [])

    return side


def _parse_switching_data(data: dict[str, Any]) -> SwitchingData:
    """Parse SwitchingData from dict."""
    sd = SwitchingData()
    sd.source = data.get('source', 'datasheet')
    sd.source_details = data.get('source_details', {})

    tc = data.get('test_conditions', {})
    sd.test_conditions = SwitchingTestConditions(
        gate_resistance_on=tc.get('gate_resistance_on', 0.0),
        gate_resistance_off=tc.get('gate_resistance_off', 0.0),
        gate_voltage_on=tc.get('gate_voltage_on', 15.0),
        gate_voltage_off=tc.get('gate_voltage_off', -5.0),
        dc_bus_voltage=tc.get('dc_bus_voltage', 0.0),
        dead_time=tc.get('dead_time', 0.0),
        parasitic_inductance=tc.get('parasitic_inductance', 0.0),
    )

    for phase_name in ['turn_on', 'turn_off', 'reverse_recovery']:
        phase_data = data.get(phase_name, {})
        phase = SwitchingEnergyData(
            temperatures=phase_data.get('temperatures', []),
            current_axis=phase_data.get('current_axis', []),
            voltage_axis=phase_data.get('voltage_axis', []),
            energy=phase_data.get('energy', {}),
        )
        setattr(sd, phase_name, phase)

    return sd
