"""Gate resistance dependent switching energy interpolation.

Recorded curves use exact physical conditions and bounded linear interpolation.
The separate scaling helper applies an explicitly requested scaling assumption.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from transistordatabase.core.models import SwitchingLossData
from transistordatabase.channel_models import real_graph
from transistordatabase.condition_tables import (
    bounded_interpolate, require_conditions, samples,
)
from transistordatabase.static_losses import finite


_NUMERIC_CONTEXT = (
    "v_supply", "v_g", "v_g_off", "load_inductance", "commutation_inductance",
)
_TEXT_CONTEXT = ("commutation_device", "convention", "event", "method")


def _nonnegative(value, label):
    """Require a finite physical magnitude."""
    value = finite(value, label)
    if value < 0:
        raise ValueError(f"{label} must be nonnegative")
    return value


def _context(record):
    """Read declared conditions without parsing comments or expressions."""
    context = {name: getattr(record, name) for name in _NUMERIC_CONTEXT}
    context["commutation_device"] = record.commutation_device
    for name in ("convention", "event", "method"):
        context[name] = getattr(record.evidence, name, None)
    return _validate_context(context)


def _validate_context(context):
    """Validate numeric and categorical conditions before comparison."""
    if set(context) - set(_NUMERIC_CONTEXT + _TEXT_CONTEXT):
        raise ValueError("Unsupported switching energy condition selector")
    for name, value in context.items():
        if value is None:
            continue
        if name in _NUMERIC_CONTEXT:
            finite(value, name)
            if name in {"load_inductance", "commutation_inductance"}:
                _nonnegative(value, name)
        elif not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be recorded text or explicitly unknown")
    return context


def _same_context(recorded, requested):
    """Match all supplied conditions, including explicit unknown values."""
    numeric = {key: value for key, value in requested.items()
               if key in _NUMERIC_CONTEXT}
    try:
        require_conditions({key: recorded[key] for key in numeric}, numeric)
    except ValueError:
        return False
    return all(recorded[key] == value for key, value in requested.items()
               if key in _TEXT_CONTEXT)


def _records(loss_data, current, temperature, conditions, representation, resistance=None):
    """Select one physical context and one applicable curve representation."""
    if representation not in {None, "graph_r_e", "graph_i_e"}:
        raise ValueError("Select graph_r_e or graph_i_e as the representation")
    requested = _validate_context(dict(conditions or {}))
    selected = []
    covered = []
    uncovered = []
    for record in loss_data:
        if record.dataset_type not in {"graph_r_e", "graph_i_e"}:
            continue
        if representation is not None and record.dataset_type != representation:
            continue
        recorded_t = finite(record.t_j, "recorded t_j")
        try:
            require_conditions({"t_j": recorded_t}, {"t_j": temperature})
        except ValueError:
            continue
        if record.dataset_type == "graph_r_e":
            if record.i_x is None:
                continue
            recorded_i = _nonnegative(record.i_x, "recorded i_x")
            try:
                require_conditions({"i_channel": recorded_i}, {"i_channel": current})
            except ValueError:
                continue
        elif resistance is not None:
            recorded_rg = _nonnegative(record.r_g, "recorded r_g")
            try:
                require_conditions({"r_g": recorded_rg}, {"r_g": resistance})
            except ValueError:
                continue
        context = _context(record)
        if not _same_context(context, requested):
            continue
        selected.append(record)
        if record.dataset_type == "graph_r_e" and resistance is None:
            covered.append((context, record))
            continue
        axis, _ = _curve(record)
        query = current if record.dataset_type == "graph_i_e" else resistance
        if axis[0] <= query <= axis[-1]:
            covered.append((context, record))
        else:
            uncovered.append((context, record))
    if not selected:
        raise ValueError("No switching energy curve matches the recorded conditions")
    if not covered:
        raise ValueError("No switching energy curve supports the requested current or Rg inside recorded coverage")
    # Numeric tolerance is not transitive. Check all applicable source pairs.
    if any(not _same_context(context, other)
           for index, (context, _) in enumerate(covered)
           for other, _ in covered[:index]):
        raise ValueError("Select one unambiguous physical switching energy context")
    result = [record for _, record in covered]
    if resistance is None:
        for context, record in uncovered:
            matches = [_same_context(context, other) for other, _ in covered]
            if any(matches) and not all(matches):
                raise ValueError("Select one unambiguous physical switching energy context")
            if all(matches):
                # Keep every Rg record in the chosen inverse context. Reject
                # current coverage holes instead of bridging omitted records.
                result.append(record)
    return result


def _curve(record):
    """Copy unique real samples and require nonnegative physical magnitudes."""
    label = record.dataset_type
    graph = real_graph(getattr(record, label), label)
    axis, energy = samples(graph[0], graph[1], label)
    if np.any(axis < 0) or np.any(energy < 0):
        raise ValueError("Resistance, current and switching energy must be nonnegative")
    return axis, energy


def interpolate_energy_at_rg(
    loss_data: list[SwitchingLossData],
    r_g: float,
    i_channel: float,
    t_j: float,
    *,
    conditions: Mapping | None = None,
    representation: str | None = None,
) -> float:
    """Estimate switching energy inside one recorded curve's coverage.

    Rg curves require the exact recorded current. Current curves require the
    exact recorded Rg. Both require exact temperature and one physical context.
    Competing applicable records require explicit selection. Missing evidence
    raises ValueError. A measured zero remains a valid result.
    The caller selects the device and switching branch through loss_data.
    Single-point records are not curves and are not used by this helper.

    :param loss_data: List of SwitchingLossData objects to search.
    :param r_g: Target gate resistance in Ohm.
    :param i_channel: Channel current in A.
    :param t_j: Junction temperature in deg C.
    :param conditions: Exact fixed-condition selectors. None matches unknown only.
        Unspecified fixed conditions must agree across the selected records.
    :param representation: Select graph_r_e or graph_i_e when both apply.
    :return: Bounded piecewise-linear switching energy estimate in J.
    :raises ValueError: If data, conditions, uniqueness or coverage is invalid.
    """
    resistance = _nonnegative(r_g, "r_g")
    current = _nonnegative(i_channel, "i_channel")
    temperature = finite(t_j, "t_j")
    matching = _records(loss_data, current, temperature, conditions, representation, resistance)
    if len(matching) != 1:
        raise ValueError("Need one unambiguous switching energy curve at exact current or Rg")
    record = matching[0]
    axis, energy = _curve(record)
    query = resistance if record.dataset_type == "graph_r_e" else current
    return bounded_interpolate(axis, energy, query, record.dataset_type)


def scale_energy_by_rg(
    e_ref: float,
    r_g_ref: float,
    r_g_target: float,
    scaling: str = "linear",
) -> float:
    """Scale a reference switching energy to a different gate resistance.

    :param e_ref: Reference switching energy in J.
    :param r_g_ref: Reference gate resistance in Ohm.
    :param r_g_target: Target gate resistance in Ohm.
    :param scaling: Scaling law to apply. ``"linear"`` uses
        ``E_target = E_ref * r_g_target / r_g_ref``.  ``"sqrt"`` uses
        ``E_target = E_ref * sqrt(r_g_target / r_g_ref)``.
    :return: Scaled switching energy in J.
    :raises ValueError: If *scaling* is not ``"linear"`` or ``"sqrt"``.
    :raises ZeroDivisionError: If *r_g_ref* is zero.
    """
    if r_g_ref == 0.0:
        raise ZeroDivisionError(
            "Reference gate resistance r_g_ref must not be zero"
        )

    if scaling == "linear":
        return e_ref * r_g_target / r_g_ref
    if scaling == "sqrt":
        return e_ref * math.sqrt(r_g_target / r_g_ref)

    raise ValueError(
        f"Unknown scaling method '{scaling}'. Use 'linear' or 'sqrt'."
    )


def find_rg_for_target_energy(
    loss_data: list[SwitchingLossData],
    e_target: float,
    i_channel: float,
    t_j: float,
    *,
    conditions: Mapping | None = None,
    representation: str | None = None,
) -> float:
    """Find one bounded piecewise-linear Rg estimate for the target energy.

    Use one exact-current Rg curve or current curves at distinct recorded Rg.
    Every current lookup must lie inside its recorded coverage. Scan all
    segments, including descending curves. Reject multiple roots and plateaus.
    The result is an interpolation estimate, not an optimal resistor rating.
    The caller selects the device and switching branch through loss_data.
    Single-point records do not define a covered resistance interval.

    :param loss_data: List of SwitchingLossData objects to search.
    :param e_target: Target switching energy in J.
    :param i_channel: Channel current in A.
    :param t_j: Junction temperature in deg C.
    :param conditions: Exact fixed-condition selectors, including explicit unknowns.
        Unspecified fixed conditions must agree across the selected records.
    :param representation: Select graph_r_e or graph_i_e when both apply.
    :return: The unique gate resistance estimate in Ohm inside recorded coverage.
    :raises ValueError: If conditions, coverage or a unique solution are absent.
    """
    target = _nonnegative(e_target, "e_target")
    current = _nonnegative(i_channel, "i_channel")
    temperature = finite(t_j, "t_j")
    candidates = _records(loss_data, current, temperature, conditions, representation)
    direct = [record for record in candidates if record.dataset_type == "graph_r_e"]
    if direct:
        if len(candidates) != 1:
            raise ValueError("Select one unambiguous switching energy curve representation")
        resistances, energies = _curve(direct[0])
    else:
        resistances, energies = [], []
        for record in candidates:
            resistances.append(_nonnegative(record.r_g, "recorded r_g"))
            axis, energy = _curve(record)
            energies.append(bounded_interpolate(axis, energy, current, "graph_i_e"))
        resistances, energies = samples(resistances, energies, "Rg-energy mapping")
    roots = [float(rg) for rg, energy in zip(resistances, energies, strict=True)
             if energy == target]
    for index in range(len(resistances) - 1):
        e0, e1 = energies[index:index + 2]
        if e0 == e1 == target:
            raise ValueError("Target energy lies on a plateau with multiple gate resistances")
        if min(e0, e1) < target < max(e0, e1):
            roots.append(bounded_interpolate(
                [e0, e1], resistances[index:index + 2], target, "Inverse Rg-energy segment",
            ))
    if len(roots) != 1:
        raise ValueError("Target energy needs one unique gate resistance inside recorded coverage")
    return roots[0]


def get_available_rg_values(
    loss_data: list[SwitchingLossData],
) -> list[float]:
    """Extract all unique gate resistance values from switching loss datasets.

    Collects ``r_g`` values that are explicitly set on the provided
    :class:`SwitchingLossData` objects.

    :param loss_data: List of SwitchingLossData objects.
    :return: Sorted list of unique Rg values.
    """
    rg_set: set[float] = set()
    for entry in loss_data:
        if entry.r_g is not None:
            rg_set.add(entry.r_g)
    return sorted(rg_set)
