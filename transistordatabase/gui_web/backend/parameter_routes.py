"""Typed transport for the shared, headless device parameter workbench."""
from __future__ import annotations

from typing import Annotated, Literal
from dataclasses import asdict
import csv
import io
import math

import numpy as np

from fastapi import APIRouter, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field, field_validator

from transistordatabase.condition_tables import (
    MeasuredTable, bounded_interpolate, separable_energy_estimate,
)
from transistordatabase.biela import fit_transconductance, extract_reverse_recovery_params
from transistordatabase.core.models import ChannelCharacteristics
from transistordatabase.device_parameters import (
    decompose_capacitances,
    energy_current_coefficient,
    fit_linear_vi,
    integrate_capacitance,
    resistance_temperature_estimate,
    signed_gate_slew,
)
from transistordatabase.gate_resistance import gate_resistance_for_slew

class ParameterRoute(APIRoute):
    """Keep validation errors JSON-safe even for nonstandard NaN request tokens."""

    def get_route_handler(self):
        """Strip raw invalid numeric inputs from the route's validation response."""
        original = super().get_route_handler()

        async def handler(request):
            try:
                return await original(request)
            except RequestValidationError as exc:
                return JSONResponse(status_code=422, content={"detail": [
                    {"loc": list(error["loc"]), "msg": error["msg"], "type": error["type"]}
                    for error in exc.errors()]})
        return handler


router = APIRouter(prefix="/api/parameters", tags=["Device parameters"],
                   route_class=ParameterRoute)
Vector = Annotated[list[float], Field(min_length=2, max_length=20000)]


class Strict(BaseModel):
    """Reject undeclared and non-finite input values."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)


class Context(Strict):
    """Recorded source and its explicit known or unknown conditions."""

    source: str = Field(min_length=1, max_length=2000)
    conditions: dict[str, float | None]

    @field_validator("source")
    @classmethod
    def nonblank_source(cls, value):
        """Require an inspectable source rather than whitespace."""
        if not value.strip():
            raise ValueError("A nonblank source identifier is required")
        return value.strip()


class Fit(Context):
    """Strict-bound linear fit input in amperes and volts."""

    operation: Literal["linear-fit"]
    current_a: Vector
    voltage_v: Vector
    lower_a: float
    upper_a: float


class Caps(Context):
    """Common-condition terminal capacitances in farads."""

    operation: Literal["terminal-capacitances"]
    c_iss_f: float
    c_oss_f: float
    c_rss_f: float


class Integration(Context):
    """Bounded capacitance-voltage integral input."""

    operation: Literal["capacitance-integration"]
    voltage_v: Vector
    capacitance_f: Vector
    start_v: float
    end_v: float


class Slew(Context):
    """Explicit constant-parameter signed slew estimate."""

    operation: Literal["gate-slew-estimate"]
    selected: bool = False
    v_th_v: float
    current_a: float
    gm_s: float
    r_g_ohm: float
    c_ies_f: float
    c_res_f: float
    v_drive_v: float


class GateResistance(Context):
    """Inverse total gate-loop resistance estimate with explicit SI inputs."""

    operation: Literal["gate-resistance-estimate"]
    selected: bool = False
    target_dv_dt_v_s: float
    c_rss_min_f: float
    v_drive_high_v: float
    v_drive_low_v: float
    v_plateau_low_v: float
    v_plateau_high_v: float


class Table(Context):
    """A single measured slice with a typed physical axis."""

    kind: Literal["energy", "energy-condition", "channel", "diode", "resistance", "r_factor"]
    x: Vector
    y: Vector

    def measured(self):
        """Construct the reusable science slice from validated transport data."""
        return MeasuredTable(**self.model_dump())


class Lookup(Strict):
    """A bounded lookup at exactly the recorded conditions."""

    operation: Literal["measured-table"]
    table: Table
    query: float
    conditions: dict[str, float | None]
    r_nom_ohm: float | None = None


class Coefficient(Context):
    """Reference energy/current ratio, kept separate from event energies."""

    operation: Literal["energy-current-coefficient"]
    energy_j: float
    reference_current_a: float


class Resistance(Context):
    """Explicit resistance-temperature and optional maximum/typical estimate."""

    operation: Literal["resistance-temperature-estimate"]
    selected: bool = False
    t1_c: float
    r1_ohm: float
    t2_c: float
    r2_ohm: float
    target_c: float
    max_typ_ratio: float | None = None


class Separable(Strict):
    """Explicit condition-sweep normalization estimate."""

    operation: Literal["separable-energy-estimate"]
    selected: bool = False
    current_table: Table
    current_a: float
    conditions: dict[str, float | None]
    slices: dict[str, Table]


class TransferConditions(Strict):
    """The actual transfer measurement condition, independent of model bus voltage."""

    t_j_c: float
    v_ds_v: float = Field(gt=0)


class TransferCurve(Strict):
    """One recorded output curve or an exact single transfer measurement."""

    v_g_v: float = Field(gt=0)
    t_j_c: float
    voltage_v: Annotated[list[float], Field(min_length=1, max_length=2000)]
    current_a: Annotated[list[float], Field(min_length=1, max_length=2000)]


class TransferExtraction(Context):
    """A bounded nonlinear transfer fit through the accepted Python helper."""

    operation: Literal["transfer-fit"]
    conditions: TransferConditions
    v_th_v: float = Field(gt=0)
    curves: Annotated[list[TransferCurve], Field(min_length=4, max_length=100)]


class RecoveryConditions(Strict):
    """Separate recovery test and single-device charge reference conditions."""

    t_j_c: float
    v_supply_v: float = Field(gt=0)
    q_oss_v_ref_v: float = Field(gt=0)
    q_oss_t_j_c: float


class RecoveryExtraction(Context):
    """Same-condition charge-control extraction in SI units."""

    operation: Literal["reverse-recovery-extraction"]
    conditions: RecoveryConditions
    q_rr_c: float = Field(gt=0)
    i_rr_a: float = Field(gt=0)
    di_dt_a_s: float = Field(gt=0)
    q_oss_c: float = Field(ge=0)
    i_forward_test_a: float = Field(gt=0)
    t_rr_s: float | None = Field(None, gt=0)


ExtractionInput = Annotated[TransferExtraction | RecoveryExtraction,
                            Field(discriminator="operation")]


class ExtractionEvidence(Strict):
    """Keep original inputs and disclose manual edits to the extracted parameters."""

    inputs: ExtractionInput
    status: Literal["applied", "edited"]


Request = Annotated[Fit | Caps | Integration | Slew | GateResistance | Lookup | Coefficient
                    | Resistance | Separable | TransferExtraction | RecoveryExtraction,
                    Field(discriminator="operation")]


def validate_biela_extraction_evidence(evidence, parameters: dict) -> list[dict]:
    """Recalculate supplied evidence and reject incompatible applied model inputs."""
    output = []
    seen = set()
    for item in evidence:
        operation = item.inputs.operation
        if operation in seen:
            raise ValueError("Provide only one extraction evidence record per operation")
        seen.add(operation)
        envelope = calculate(item.inputs)
        result = envelope["result"]
        expected = result["biela_parameters"]
        if item.status == "applied":
            if any(parameters.get(key) != value for key, value in expected.items()):
                raise ValueError("Applied extraction parameters changed. Mark their evidence as edited or reapply the extraction")
            condition = item.inputs.conditions
            if parameters["parameter_t_j"] != condition.t_j_c:
                raise ValueError("Applied extraction temperature must match the model parameter temperature")
            if isinstance(item.inputs, RecoveryExtraction):
                if (parameters["recovery_mode"] != "charge_control"
                        or parameters["q_oss"] != item.inputs.q_oss_c
                        or parameters["q_oss_v_ref"] != condition.q_oss_v_ref_v
                        or parameters["v_dc"] != condition.v_supply_v):
                    raise ValueError("Recovery source Qoss, reference voltage and temperature must match the model. Supply compatible model inputs")
        output.append({"status": item.status, **envelope})
    return output


def _extraction_csv(request, result: dict) -> str:
    """Export authoritative SI rows and spreadsheet-safe source text."""
    from transistordatabase.digitization import _csv_text
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    source = _csv_text(request.source)
    condition = request.conditions
    if isinstance(request, TransferExtraction):
        columns = ["gate_voltage_v", "current_a", "fitted_current_a", "residual_a", "secant_gm_s"]
        writer.writerow(columns + ["source", "v_ds_v", "t_j_c", "v_th_v"])
        for row in zip(*(result[key] for key in columns), strict=True):
            writer.writerow([*row, source, condition.v_ds_v, condition.t_j_c, request.v_th_v])
    else:
        inputs = ["q_rr_c", "i_rr_a", "di_dt_a_s", "q_oss_c", "i_forward_test_a", "t_rr_s"]
        outputs = ["q_rr_star", "tau_rr", "tau_c", "t_m", "q_rf_c", "t_1_s",
                   "peak_residual_a", "lifetime_residual_per_s"]
        conditions = ["v_supply_v", "t_j_c", "q_oss_v_ref_v", "q_oss_t_j_c"]
        writer.writerow(inputs + [key + "_s" if key in {"tau_rr", "tau_c", "t_m"}
                                  else key for key in outputs] + ["source"] + conditions)
        writer.writerow([*[getattr(request, key) for key in inputs],
                         *[result[key] for key in outputs], source,
                         *[getattr(condition, key) for key in conditions]])
    return stream.getvalue()


@router.post("/calculate")
def calculate(request: Request) -> dict:
    """Return auditable inputs, SI results and plot samples from one calculation."""
    values = request.model_dump()
    operation = values.pop("operation")
    context = {key: values.pop(key) for key in ("source", "conditions") if key in values}
    plots = []
    try:
        if isinstance(request, TransferExtraction):
            condition = request.conditions
            if sum(len(curve.voltage_v) for curve in request.curves) > 10000:
                raise ValueError("Transfer extraction supports at most 10000 recorded points")
            curves, recorded = [], []
            for curve in request.curves:
                if curve.t_j_c != condition.t_j_c:
                    raise ValueError("Every transfer curve must have the exact declared temperature")
                if len(curve.voltage_v) != len(curve.current_a):
                    raise ValueError("Each transfer curve requires matching voltage and current columns")
                graph = np.array([curve.voltage_v, curve.current_a])
                curves.append(ChannelCharacteristics(curve.t_j_c, graph, curve.v_g_v))
                current = (curve.current_a[0] if len(curve.voltage_v) == 1 else
                           bounded_interpolate(curve.voltage_v, curve.current_a,
                                               condition.v_ds_v, label="transfer Vds"))
                recorded.append((curve.v_g_v, float(current)))
            fitted = fit_transconductance(curves, v_ds=condition.v_ds_v,
                                          t_j=condition.t_j_c, v_th=request.v_th_v)
            recorded.sort()
            gate, current = map(list, zip(*recorded, strict=True))
            prediction = [fitted.k_1 * (v - fitted.v_th) ** fitted.x + fitted.k_2 for v in gate]
            residual = [p - i for p, i in zip(prediction, current, strict=True)]
            gm = [fitted.calc_gm(i) for i in current]
            if not all(math.isfinite(v) for v in prediction + residual + gm):
                raise ValueError("Transfer result exceeds the finite numerical range")
            result = {**asdict(fitted), "gate_voltage_v": gate, "current_a": current,
                      "fitted_current_a": prediction, "residual_a": residual,
                      "secant_gm_s": gm,
                      "biela_parameters": {"k_1": fitted.k_1, "k_2": fitted.k_2,
                                           "x": fitted.x, "v_th": fitted.v_th,
                                           "gm_i_min": fitted.current_min,
                                           "gm_i_max": fitted.current_max},
                      "scope": "Nonlinear transfer fit at the recorded Vds/Tj; no model bus-voltage equivalence or measured accuracy claim",
                      "gm_definition": "Secant I/(Vgs-Vth), not the differential derivative"}
            plots = [{"title": "Transfer fit at recorded Vds/Tj", "x_label": "Gate voltage (V)",
                      "y_label": "Channel current (A)", "x": gate,
                      "series": {"Recorded": current, "Fitted": prediction}},
                     {"title": "Transfer residual", "x_label": "Gate voltage (V)",
                      "y_label": "Fitted minus recorded current (A)", "x": gate,
                      "series": {"Residual": residual}}]
        elif isinstance(request, RecoveryExtraction):
            condition = request.conditions
            if (condition.v_supply_v != condition.q_oss_v_ref_v
                    or condition.t_j_c != condition.q_oss_t_j_c):
                raise ValueError("Recovery Qoss must have the same test voltage and temperature as Qrr")
            recovery = extract_reverse_recovery_params(
                request.q_rr_c, request.i_rr_a, request.di_dt_a_s, request.q_oss_c,
                request.t_rr_s, i_forward_test=request.i_forward_test_a)
            t1 = (request.i_forward_test_a + request.i_rr_a) / request.di_dt_a_s
            predicted_peak = (request.di_dt_a_s * (recovery.tau_c - recovery.tau_rr)
                              * -math.expm1(-t1 / recovery.tau_c))
            result = {**asdict(recovery), "biela_parameters": asdict(recovery),
                      "q_rf_c": recovery.tau_rr * request.i_rr_a, "t_1_s": t1,
                      "peak_residual_a": predicted_peak - request.i_rr_a,
                      "lifetime_residual_per_s": 1 / recovery.t_m - (
                          1 / recovery.tau_rr - 1 / recovery.tau_c),
                      "t_rr_used": False,
                      "scope": "Equations 38–41 charge-control extraction; infinite exponential tail. Trr is compatibility metadata only",
                      "equations": ["Q*=Qrr-Qoss", "Qrf=Q*-Irr^2/(2*di_dt)",
                                    "tau_rr=Qrf/Irr", "1/Tm=1/tau_rr-1/tau_c"],
                      "accuracy": "Calculated model parameters, not measured device-accuracy evidence"}
            numeric = [value for value in result.values() if isinstance(value, (int, float))]
            if not all(math.isfinite(v) for v in numeric):
                raise ValueError("Recovery result exceeds the finite numerical range")
        elif isinstance(request, Fit):
            result = fit_linear_vi(**values)
            plots = [{"title": "Linear V-I fit", "x_label": "Current (A)",
                      "y_label": "Voltage (V)", "x": result["current_a"],
                      "series": {"Recorded": result["voltage_v"],
                                 "Fitted": result["fitted_voltage_v"]}}]
        elif isinstance(request, Caps):
            result = decompose_capacitances(**values)
        elif isinstance(request, Integration):
            result = integrate_capacitance(**values)
            plots = [{"title": "Bounded capacitance", "x_label": "Voltage (V)",
                      "y_label": "Capacitance (F)", "x": result["voltage_v"],
                      "series": {"C(V)": result["capacitance_f"]}},
                     {"title": "Integrated charge", "x_label": "Voltage (V)",
                      "y_label": "Charge from start (C)", "x": result["voltage_v"],
                      "series": {"Q": result["q_c"]}}]
        elif isinstance(request, Slew):
            if not values.pop("selected"):
                raise ValueError("Select the constant-gm/capacitance slew estimate explicitly")
            result = signed_gate_slew(**values)
            result["scope"] = "Constant gm and capacitance estimate; no parasitic transient model"
        elif isinstance(request, GateResistance):
            result = gate_resistance_for_slew(**values)
        elif isinstance(request, Lookup):
            table = request.table.measured()
            if table.kind == "energy-condition":
                raise ValueError("Condition sweeps are evaluated within the separable estimate")
            result = {"value": table.evaluate(request.query, request.conditions,
                                               request.r_nom_ohm),
                      "mode": "bounded-measured-slice", "kind": table.kind,
                      "source": table.source, "recorded_conditions": dict(table.conditions),
                      "coverage": [float(table.x[0]), float(table.x[-1])]}
            result.update({"unit": {"energy": "J", "channel": "V", "diode": "V",
                                    "resistance": "ohm", "r_factor": "ohm"}[table.kind],
                           "query_unit": "degC" if table.kind in {"resistance", "r_factor"} else "A",
                           "query": request.query})
            if table.kind == "r_factor":
                result.update(r_nom_ohm=request.r_nom_ohm,
                              factor=table.evaluate(request.query, request.conditions, 1.0),
                              factor_unit="1")
            units = {"energy": ("Current (A)", "Energy (J)"),
                     "channel": ("Current (A)", "Voltage (V)"),
                     "diode": ("Current (A)", "Voltage (V)"),
                     "resistance": ("Temperature (°C)", "Resistance (Ω)"),
                     "r_factor": ("Temperature (°C)", "Resistance factor (1)")}
            xlabel, ylabel = units[table.kind]
            plots = [{"title": "Recorded bounded slice", "x_label": xlabel,
                      "y_label": ylabel, "x": table.x.tolist(),
                      "series": {"Recorded": table.y.tolist()}}]
        elif isinstance(request, Coefficient):
            result = energy_current_coefficient(**values, **context)
        elif isinstance(request, Resistance):
            result = resistance_temperature_estimate(**values)
        else:
            result = separable_energy_estimate(
                request.current_table.measured(), request.current_a, request.conditions,
                {name: table.measured() for name, table in request.slices.items()},
                selected=request.selected)
        output = {"operation": operation, "inputs": request.model_dump(),
                "context": context, "result": result, "plots": plots,
                "units": "SI; temperature in degrees Celsius",
                "condition_policy": "Exact recorded conditions; null means unknown, not wildcard"}
        if isinstance(request, (TransferExtraction, RecoveryExtraction)):
            output["csv"] = _extraction_csv(request, result)
        return output
    except (ValueError, TypeError, OverflowError, FloatingPointError, ZeroDivisionError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
