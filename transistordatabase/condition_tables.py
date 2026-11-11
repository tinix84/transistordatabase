"""Bounded one-dimensional data and explicit separable estimates, in SI units.

Conditions are matched exactly (within 1 n-unit numeric rounding). An unknown
condition is represented by None and never matches a specified condition.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np


def samples(x: Sequence[float], y: Sequence[float], label: str = "table"):
    """Validate, sort and copy unique finite paired samples."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 2:
        raise ValueError(f"{label} needs at least two aligned sample pairs")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f"{label} samples must be finite")
    order = np.argsort(x, kind="stable")
    x, y = x[order].copy(), y[order].copy()
    if np.any(np.diff(x) <= 0):
        raise ValueError(f"{label} needs unique x samples")
    return x, y


def bounded_interpolate(x, y, query: float, label: str = "table") -> float:
    """Piecewise linear interpolation with no clamping or extrapolation."""
    x, y = samples(x, y, label)
    return _bounded_sorted(x, y, query, label)


def _bounded_sorted(x, y, query: float, label: str = "table") -> float:
    """Look up samples already validated/copy-frozen by this module."""
    query = float(query)
    if not np.isfinite(query) or not x[0] <= query <= x[-1]:
        raise ValueError(f"{label} supports {x[0]:g} to {x[-1]:g}; requested {query:g}")
    result = float(np.interp(query, x, y))
    if not np.isfinite(result):
        raise ValueError(f"{label} interpolation overflow")
    return result


def same_condition(left, right) -> bool:
    """Compare finite recorded numeric conditions without treating unknown as zero."""
    for value in (left, right):
        if value is not None and (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)):
            raise ValueError("Conditions must be real numeric scalars or explicitly unknown")
    if left is None or right is None:
        return left is right
    if not np.isfinite(left) or not np.isfinite(right):
        raise ValueError("Conditions must be finite or explicitly unknown")
    return bool(np.isclose(left, right, rtol=0, atol=1e-9))


def require_conditions(recorded: Mapping, requested: Mapping) -> None:
    """Require the same named conditions, including explicit unknowns."""
    if recorded.keys() != requested.keys() or any(
        not same_condition(value, requested[key]) for key, value in recorded.items()
    ):
        raise ValueError("Requested conditions do not match the recorded conditions")


@dataclass(frozen=True)
class MeasuredTable:
    """A measured slice; temperature and voltage are not interpolated implicitly."""

    kind: str
    x: Sequence[float]
    y: Sequence[float]
    conditions: Mapping[str, float | None]
    source: str

    def __post_init__(self):
        """Copy and validate the supplied slice once, preserving its source."""
        if self.kind not in {"energy", "energy-condition", "channel", "diode", "resistance", "r_factor"}:
            raise ValueError("Unsupported measured table kind")
        if not self.source.strip():
            raise ValueError("A source identifier is required")
        x, y = samples(self.x, self.y)
        if np.any(y < 0) or (self.kind in {"energy", "diode", "channel"} and np.any(x < 0)):
            raise ValueError("Energy/diode currents and table values must be nonnegative")
        if self.kind in {"resistance", "r_factor"} and np.any(y <= 0):
            raise ValueError("Resistance and resistance factors must be positive")
        for value in self.conditions.values():
            same_condition(value, value)
        x.flags.writeable = y.flags.writeable = False
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "conditions", MappingProxyType(dict(self.conditions)))

    def evaluate(self, query: float, conditions: Mapping, r_nom_ohm=None) -> float:
        """Evaluate at matching conditions; factors additionally require Rnom."""
        require_conditions(self.conditions, conditions)
        value = _bounded_sorted(self.x, self.y, query, self.kind)
        if self.kind == "r_factor":
            if r_nom_ohm is None or not np.isfinite(r_nom_ohm) or r_nom_ohm <= 0:
                raise ValueError("R(T) factors require a finite positive Rnom in ohms")
            value *= r_nom_ohm
        if not np.isfinite(value):
            raise ValueError("Table result overflow")
        return float(value)


def separable_energy_estimate(
    current_table: MeasuredTable, current_a: float, requested: Mapping,
    slices: Mapping[str, MeasuredTable], *, selected: bool = False,
) -> dict:
    """Scale a current slice by normalized, bounded isolated condition sweeps.

    This assumes independent multiplicative effects. It is never a measured
    multidimensional surface. Each sweep's other conditions must match baseline.
    """
    if not selected:
        raise ValueError("Select the separable energy estimate explicitly")
    if current_table.kind != "energy":
        raise ValueError("The baseline must be an energy-current table")
    baseline = dict(current_table.conditions)
    if baseline.keys() != requested.keys():
        raise ValueError("Requested conditions must name every baseline condition")
    value = current_table.evaluate(current_a, baseline)
    factors, diagnostics = {}, []
    for name, target in requested.items():
        if same_condition(baseline[name], target):
            continue
        if target is None or baseline[name] is None or name not in slices:
            raise ValueError(f"No bounded measured condition sweep for {name}")
        sweep = slices[name]
        if sweep.kind != "energy-condition":
            raise ValueError(f"{name} sweep must contain condition-axis energies")
        other = {key: val for key, val in baseline.items() if key != name}
        # A sweep is measured at a specified reference current, supplied separately.
        if "current_a" not in sweep.conditions:
            raise ValueError(f"{name} sweep needs its reference current_a")
        reference_current = sweep.conditions["current_a"]
        require_conditions(other, {k: v for k, v in sweep.conditions.items()
                                   if k != "current_a"})
        norm = bounded_interpolate(sweep.x, sweep.y, baseline[name], name)
        target_energy = bounded_interpolate(sweep.x, sweep.y, target, name)
        baseline_energy = current_table.evaluate(reference_current, baseline)
        if norm <= 0 or baseline_energy <= 0:
            raise ValueError("Separable normalization requires positive reference energy")
        factor = target_energy / norm
        factors[name] = float(factor)
        value *= factor
        diagnostics.append({"condition": name, "source": sweep.source,
                            "baseline_residual_j": float(norm - baseline_energy),
                            "coverage": [float(sweep.x[0]), float(sweep.x[-1])],
                            "reference_current_a": reference_current})
    if not np.isfinite(value):
        raise ValueError("Separable estimate overflow")
    return {"energy_j": float(value), "mode": "separable-energy-estimate",
            "factors": factors, "slice_diagnostics": diagnostics,
            "source": current_table.source, "baseline_conditions": baseline,
            "requested_conditions": dict(requested),
            "assumptions": ["Independent multiplicative condition effects",
                            "No measured multidimensional surface or interaction validation"]}
