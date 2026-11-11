"""Exact channel selection and bounded local V = V0 + R I models."""
from __future__ import annotations

import numpy as np

from transistordatabase.condition_tables import (
    bounded_interpolate, require_conditions, samples,
)
from transistordatabase.static_losses import finite


def real_graph(graph, label):
    """Require two rows of finite real source numbers before float conversion."""
    source = np.asarray(graph)
    if source.dtype.kind not in "iuf":
        raise ValueError(f"{label} requires real numeric source samples")
    if source.ndim != 2 or source.shape[0] != 2:
        raise ValueError(f"{label} needs two aligned rows")
    converted = np.asarray(source, dtype=float)
    if not np.all(np.isfinite(converted)):
        raise ValueError(f"{label} requires finite source samples")
    return converted


def channel_records(records, t_j, v_g):
    """Select recorded conditions exactly. None means an unknown gate voltage."""
    t_j = finite(t_j, "t_j")
    if v_g is not None:
        v_g = finite(v_g, "v_g")
    requested = {"t_j": t_j, "v_g": v_g}
    selected = []
    for record in records:
        recorded = {"t_j": record.t_j, "v_g": record.v_g}
        # Invalid stored evidence is an error, distinct from an absent slice.
        for name, value in recorded.items():
            if value is not None:
                finite(value, f"recorded {name}")
        try:
            require_conditions(recorded, requested)
        except ValueError:
            continue
        selected.append(record)
    return selected


def local_channel_model(records, t_j, v_g, i_channel):
    """Use the containing segment, with right derivatives at interior knots.

    Endpoints use the one-sided measured segment. Signed samples remain signed.
    No nearest condition, extrapolation, or least-squares fit is implied.
    """
    selected = channel_records(records, t_j, v_g)
    if len(selected) != 1:
        raise ValueError("Need one unambiguous channel record at exact Tj and gate voltage")
    graph = real_graph(selected[0].graph_v_i, "Channel V(I)")
    current, voltage = samples(graph[1], graph[0], "Channel V(I)")
    operating_current = finite(i_channel, "i_channel")
    operating_voltage = bounded_interpolate(current, voltage, operating_current, "Channel V(I)")
    index = min(max(int(np.searchsorted(current, operating_current, side="right")) - 1, 0), len(current) - 2)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        span = current[index + 1] - current[index]
        change = voltage[index + 1] - voltage[index]
        resistance = change / span
        offset = operating_voltage - resistance * operating_current
    if (not np.isfinite(span) or not np.isfinite(change)
            or not np.isfinite(resistance) or not np.isfinite(offset)
            or (change != 0 and resistance == 0)):
        raise ValueError("Channel linearization exceeds finite numeric range")
    return float(resistance), float(offset)
