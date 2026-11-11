"""Stateless, bounded semiconductor fits with explicit source observations.

One scaled polynomial method is used throughout. The SVD condition limit is
1e8. Diode slopes must be strictly positive throughout the fitted interval;
this conservative rule excludes flat segments. No fit uses holdout rows.
"""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import hashlib
import json
from numbers import Real

import numpy as np

METHOD = "scaled-polynomial-device-fit-v1"
SCOPE = ("Bounded polynomial estimate from supplied observations; SI units with "
         "temperature in degC. No extrapolation or automatic database changes. "
         "Holdout rows from shared sources are not independent measurements. "
         "Diode current and voltage are positive magnitudes; slopes are strictly "
         "positive. Energy assumes separable isolated sweeps.")
AXES = ("current_a", "voltage_v", "t_j_c", "r_g_ohm")
UNITS = {"resistance": "ohm", "voltage": "V", "current": "A", "energy": "J"}
AXIS_UNITS = dict(zip(AXES, ("A", "V", "degC", "ohm"), strict=True))
MAX_ROWS = 2000
MAX_SLICES = 32


def _number(value, label, nullable=False):
    if nullable and value is None:
        return None
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (Real, Decimal))):
        raise ValueError(f"{label} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{label} exceeds the finite numerical range") from exc
    if not np.isfinite(result) or (result == 0 and value != 0):
        raise ValueError(f"{label} exceeds the finite numerical range")
    return result


def _keys(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"{label} requires exactly these fields: {', '.join(keys)}")


def _text(value, label):
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"{label} needs 1 to 256 nonblank characters")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{label} must contain valid UTF-8 scalar text") from exc
    return value


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False, ensure_ascii=False).encode("utf-8")
    except (ValueError, TypeError, OverflowError, UnicodeEncodeError,
            RecursionError) as exc:
        raise ValueError("Supply finite JSON data within the parser limits") from exc


def _hash(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _bounded(value):
    if len(_canonical(value)) > 4 * 1024 * 1024:
        raise ValueError("Input exceeds the 4 MiB limit")


def _bounded_result(value):
    if len(_canonical(value)) > 24 * 1024 * 1024:
        raise ValueError("Result exceeds the 24 MiB limit; reduce supplied observations")
    return value


def _coordinates(value, keys, label, nullable=()):
    _keys(value, keys, label)
    result = {key: _number(value[key], key, key == "v_g_v" or key in nullable)
              for key in keys}
    for key in ("current_a", "voltage_v", "r_g_ohm"):
        if key in result and result[key] is not None and result[key] < 0:
            raise ValueError(f"{key} must be nonnegative; use positive magnitudes")
    return result


def _inputs(request):
    if not isinstance(request, dict):
        raise ValueError("Supply a training request object")
    request = {"degree": 1, "event": None, "fixed_conditions": {},
               "anchor": None, **request}
    _keys(request, ("family", "degree", "event", "fixed_conditions", "anchor",
                    "observations"), "Training request")
    family, degree = request["family"], request["degree"]
    if family not in ("ron", "diode", "energy"):
        raise ValueError("family must be ron, diode or energy")
    if type(degree) is not int or degree not in (1, 2, 3):
        raise ValueError("degree must be the integer 1, 2 or 3")
    fixed_keys = {"ron": ("v_g_v", "current_a"), "diode": (),
                  "energy": ("v_g_v",)}[family]
    _keys(request["fixed_conditions"], fixed_keys, "Fixed conditions")
    fixed = {key: _number(request["fixed_conditions"][key], key, True)
             for key in fixed_keys}
    if fixed.get("current_a") is not None and fixed["current_a"] < 0:
        raise ValueError("Fixed current must be nonnegative")
    if family == "energy":
        if request["event"] not in ("e_on", "e_off"):
            raise ValueError("Energy fits require one explicit e_on or e_off event")
        anchor = _coordinates(request["anchor"], AXES, "Energy anchor")
        if anchor["current_a"] <= 0 or anchor["voltage_v"] <= 0:
            raise ValueError("Energy nominal current and voltage must be positive")
    else:
        if request["event"] is not None or request["anchor"] is not None:
            raise ValueError("Only energy fits accept an event and anchor")
        anchor = None
    rows = request["observations"]
    if not isinstance(rows, list) or not 2 <= len(rows) <= MAX_ROWS:
        raise ValueError(f"Supply 2 to {MAX_ROWS} observations")
    keys = {"ron": ("t_j_c",), "diode": ("current_a", "t_j_c", "v_g_v"),
            "energy": AXES}[family]
    normalized, ids = [], set()
    for row in rows:
        _keys(row, ("id", "source", "series", "split", "coordinates", "value"),
              "Observation")
        item = {key: _text(row[key], key) for key in ("id", "source", "series")}
        if item["id"] in ids:
            raise ValueError("Observation IDs must be unique")
        ids.add(item["id"])
        if row["split"] not in ("train", "holdout"):
            raise ValueError("Observation split must be train or holdout")
        item.update(split=row["split"], coordinates=_coordinates(
            row["coordinates"], keys, "Observation coordinates",
            nullable=("t_j_c",) if family == "diode" else ()),
            value=_number(row["value"], "Observed value"))
        if item["value"] < 0 or (family == "ron" and item["value"] == 0):
            raise ValueError("Observed resistance must be positive; other values nonnegative")
        normalized.append(item)
    return dict(family=family, degree=degree, event=request["event"],
                fixed_conditions=fixed, anchor=anchor, observations=normalized)


def _critical(coefficients, lower, upper, derivative=1):
    # Derivative roots are invariant under positive coefficient scaling. Scale
    # first so differentiating a finite large coefficient cannot overflow.
    magnitude = max(abs(value) for value in coefficients)
    normalized = np.asarray(coefficients) / magnitude if magnitude else coefficients
    derived = np.polynomial.polynomial.polyder(normalized, derivative)
    try:
        roots = np.polynomial.polynomial.polyroots(derived)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Polynomial extrema cannot be resolved safely") from exc
    return [float(root.real) for root in roots
            if abs(root.imag) < 1e-10 and lower < root.real < upper]


def _physical(poly, positive=False, monotone=False):
    coefficients = poly["coefficients"]
    if not np.isfinite(coefficients).all():
        raise ValueError("Fitted coefficients exceed the finite numerical range")
    lower, upper = [(value - poly["center"]) / poly["scale"]
                    for value in poly["support"]]
    points = [lower, upper, *_critical(coefficients, lower, upper)]
    values = np.polynomial.polynomial.polyval(points, coefficients)
    if (not np.isfinite(values).all() or np.any(values < 0)
            or (positive and np.any(values <= 0))):
        raise ValueError("Fitted polynomial violates physical values inside its support")
    if monotone:
        points = [lower, upper, *_critical(coefficients, lower, upper, 2)]
        magnitude = max(abs(value) for value in coefficients)
        normalized = np.asarray(coefficients) / magnitude if magnitude else coefficients
        slopes = np.polynomial.polynomial.polyval(
            points, np.polynomial.polynomial.polyder(normalized))
        if not np.isfinite(slopes).all() or np.any(slopes <= 0):
            raise ValueError("Diode polynomial must have strictly positive slope throughout support")


def _poly(rows, axis, degree, inverse=False, positive=False, monotone=False):
    x = np.array([row["value"] if inverse else row["coordinates"][axis]
                  for row in rows], dtype=float)
    y = np.array([row["coordinates"]["current_a"] if inverse else row["value"]
                  for row in rows], dtype=float)
    if len(np.unique(x)) < degree + 1:
        raise ValueError(f"{axis} needs at least degree + 1 distinct training coordinates")
    lo, hi = float(x.min()), float(x.max())
    center, scale = lo / 2 + hi / 2, hi / 2 - lo / 2
    if not np.isfinite([center, scale]).all() or scale <= 0:
        raise ValueError("Training coordinate range cannot be scaled safely")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        z = (x - center) / scale
        matrix = np.polynomial.polynomial.polyvander(z, degree)
    if not np.isfinite(matrix).all():
        raise ValueError("Training coordinates exceed the numerical range")
    try:
        coefficients, _, rank, singular = np.linalg.lstsq(matrix, y, rcond=None)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Polynomial least squares did not converge") from exc
    condition = float(singular[0] / singular[-1])
    if rank != degree + 1 or not np.isfinite(condition) or condition > 1e8:
        raise ValueError("Polynomial design is rank deficient or exceeds condition limit 1e8")
    if not np.isfinite(coefficients).all():
        raise ValueError("Fitted coefficients exceed the finite numerical range")
    poly = dict(basis="ascending-powers-of-z", degree=degree,
                coefficients=coefficients.tolist(), center=center, scale=scale,
                support=[lo, hi], training_ids=[row["id"] for row in rows],
                rank=int(rank), condition_number=condition,
                residual_degrees_of_freedom=len(rows) - int(rank), normalization=None)
    _physical(poly, positive, monotone)
    return poly


def _value(poly, coordinate):
    if not poly["support"][0] <= coordinate <= poly["support"][1]:
        raise ValueError(f"Query exceeds fitted support {poly['support']}")
    result = float(np.polynomial.polynomial.polyval(
        (coordinate - poly["center"]) / poly["scale"], poly["coefficients"]))
    if not np.isfinite(result):
        raise ValueError("Prediction exceeds the finite numerical range")
    return result


def _build(inputs):
    family, degree = inputs["family"], inputs["degree"]
    rows = [row for row in inputs["observations"] if row["split"] == "train"]
    slices = []
    if family == "ron":
        slices.append(dict(id="ron", conditions=inputs["fixed_conditions"],
                           polynomials={"resistance": _poly(
                               rows, "t_j_c", degree, positive=True)}))
    elif family == "diode":
        groups = {}
        for row in rows:
            coordinates = row["coordinates"]
            key = (coordinates["t_j_c"], coordinates["v_g_v"])
            groups.setdefault(key, []).append(row)
        if not groups or len(groups) > MAX_SLICES:
            raise ValueError(f"Supply 1 to {MAX_SLICES} exact temperature/gate slices")
        for index, ((temperature, gate), group) in enumerate(groups.items()):
            slices.append(dict(id=f"diode-{index + 1}",
                               conditions={"t_j_c": temperature, "v_g_v": gate},
                               polynomials={
                                   "voltage": _poly(group, "current_a", degree,
                                                    monotone=True),
                                   "current": _poly(group, "voltage_v", degree,
                                                    inverse=True, monotone=True)}))
    else:
        anchor = inputs["anchor"]
        for row in rows:
            changed = [axis for axis in AXES
                       if row["coordinates"][axis] != anchor[axis]]
            if len(changed) > 1:
                raise ValueError("Energy training rows must be isolated nominal-axis sweeps")
        polynomials = {}
        for axis in AXES:
            group = [row for row in rows if all(
                row["coordinates"][other] == anchor[other]
                for other in AXES if other != axis)]
            if axis != "current_a" and not any(
                    row["coordinates"][axis] != anchor[axis] for row in rows):
                continue
            poly = _poly(group, axis, degree)
            nominal = _value(poly, anchor[axis])
            if nominal <= 0:
                raise ValueError("Fitted energy at each nominal anchor must be positive")
            if axis != "current_a":
                poly["coefficients"] = [value / nominal
                                        for value in poly["coefficients"]]
                poly["normalization"] = nominal
                _physical(poly)
            polynomials[axis] = poly
        slices.append(dict(id=inputs["event"], conditions=inputs["fixed_conditions"],
                           polynomials=polynomials))
    return deepcopy(dict(schema_version=1, method=METHOD, family=family, degree=degree,
                         event=inputs["event"], fixed_conditions=inputs["fixed_conditions"],
                         anchor=inputs["anchor"], training_observations=rows,
                         slices=slices))


def _same_json(left, right):
    """Compare JSON numbers by value, while keeping booleans distinct from numbers."""
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _same_json(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _same_json(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def validate_device_model(model):
    """Validate an untrusted JSON model by reproducing its declared training fit.

    This checks all coefficients, source rows, support, rank, physical bounds and
    method metadata. Model documents contain data only, never executable state.
    """
    _bounded(model)
    _keys(model, ("schema_version", "method", "family", "degree", "event",
                  "fixed_conditions", "anchor", "training_observations", "slices"),
          "Model")
    if type(model["schema_version"]) is not int or model["schema_version"] != 1:
        raise ValueError("Unsupported model schema version")
    if model["method"] != METHOD:
        raise ValueError("Unsupported model method")
    inputs = _inputs({key: model[key] for key in
                      ("family", "degree", "event", "fixed_conditions", "anchor")}
                     | {"observations": model["training_observations"]})
    if any(row["split"] != "train" for row in inputs["observations"]):
        raise ValueError("Model training observations cannot contain holdout rows")
    rebuilt = _build(inputs)
    if not _same_json(model, rebuilt):
        raise ValueError("Model state does not match its validated training observations")
    return rebuilt


def _predict(model, output, coordinates):
    family = model["family"]
    expected = {"ron": ("resistance",), "diode": ("voltage", "current"),
                "energy": ("energy",)}[family]
    if output not in expected:
        raise ValueError("Requested output does not belong to this model family")
    if family == "ron":
        c = _coordinates(coordinates, ("t_j_c", "v_g_v", "current_a"),
                         "Resistance query coordinates", nullable=("current_a",))
        # An explicitly unknown fixed current is a valid condition, not zero.
        if any(c[key] != value for key, value in model["fixed_conditions"].items()):
            raise ValueError("Query conditions do not match the recorded fixed conditions")
        return _value(model["slices"][0]["polynomials"]["resistance"], c["t_j_c"])
    if family == "diode":
        axis = "current_a" if output == "voltage" else "voltage_v"
        c = _coordinates(coordinates, (axis, "t_j_c", "v_g_v"),
                         "Diode query coordinates", nullable=("t_j_c",))
        match = [item for item in model["slices"] if item["conditions"] == {
            "t_j_c": c["t_j_c"], "v_g_v": c["v_g_v"]}]
        if len(match) != 1:
            raise ValueError("No exact recorded temperature/gate slice matches the query")
        return _value(match[0]["polynomials"][output], c[axis])
    c = _coordinates(coordinates, (*AXES, "v_g_v"), "Energy query coordinates")
    if c["v_g_v"] != model["fixed_conditions"]["v_g_v"]:
        raise ValueError("Query gate does not match the recorded fixed condition")
    result = 1.0
    for axis in AXES:
        poly = model["slices"][0]["polynomials"].get(axis)
        if poly is None:
            if c[axis] != model["anchor"][axis]:
                raise ValueError(f"Unfitted {axis} is fixed at its nominal anchor")
        else:
            result *= _value(poly, c[axis])
    if not np.isfinite(result):
        raise ValueError("Energy product exceeds the finite numerical range")
    return result


def _observation_query(model, row, output):
    c = dict(row["coordinates"])
    if model["family"] == "ron":
        c.update(model["fixed_conditions"])
    elif model["family"] == "diode" and output == "current":
        c.pop("current_a")
        c["voltage_v"] = row["value"]
    elif model["family"] == "energy":
        c.update(model["fixed_conditions"])
    return c


def _metrics(rows):
    scored = [row for row in rows if row["status"] == "predicted"]
    residuals = np.array([row["residual"] for row in scored])
    observed = np.array([row["observed"] for row in scored])
    # Scale before squaring to avoid overflow for finite large observations.
    if len(scored):
        scale = float(max(abs(residuals).max(), 1e-300))
        rmse = float(np.sqrt(np.mean((residuals / scale) ** 2)) * scale)
        mae = float(np.mean(np.abs(residuals) / scale) * scale)
    else:
        rmse = mae = None
    r2 = None
    if len(scored) >= 2 and observed.min() != observed.max():
        # Subtract one observed origin first: nearby values at large offsets
        # must retain their variation rather than round into a constant series.
        shifted = observed - observed[0]
        span = float(observed.max() - observed.min())
        normalized = shifted / span
        centered = normalized - np.mean(normalized)
        denominator = float(np.sum(centered ** 2))
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            candidate = 1 - float(np.sum((residuals / span) ** 2)) / denominator
        if not np.isfinite(candidate):
            raise ValueError("R2 exceeds the finite numerical range")
        r2 = candidate
    return dict(rows=rows, count=len(scored), rmse=rmse, mae=mae, r2=r2)


def _diagnostics(model, observations):
    diagnostics = []
    outputs = {"ron": ["resistance"], "diode": ["voltage", "current"],
               "energy": ["energy"]}[model["family"]]
    for slice_ in model["slices"]:
        for output in outputs:
            splits = {"train": [], "holdout": []}
            for row in observations:
                if model["family"] == "diode":
                    conditions = {key: row["coordinates"][key]
                                  for key in ("t_j_c", "v_g_v")}
                    if conditions != slice_["conditions"]:
                        continue
                observed = (row["coordinates"]["current_a"]
                            if output == "current" else row["value"])
                item = {key: row[key] for key in ("id", "source", "series")}
                item.update(observed=observed, predicted=None, residual=None,
                            status="rejected", reason=None)
                try:
                    predicted = _predict(model, output, _observation_query(model, row, output))
                    residual = observed - predicted
                    if not np.isfinite(residual):
                        raise ValueError("Residual exceeds the finite numerical range")
                    item.update(predicted=predicted, residual=residual, status="predicted")
                except ValueError as exc:
                    item["reason"] = str(exc)
                splits[row["split"]].append(item)
            diagnostics.append(dict(output=output, slice=slice_["id"],
                                    train=_metrics(splits["train"]),
                                    holdout=_metrics(splits["holdout"])
                                    if splits["holdout"] else None))
    # Keep unmatched held-out diode slices visible, rather than silently dropping them.
    if model["family"] == "diode":
        unmatched = [row for row in observations if row["split"] == "holdout"
                     and not any(all(row["coordinates"][key] == value
                                     for key, value in item["conditions"].items())
                                 for item in model["slices"])]
        if unmatched:
            for output in outputs:
                rows = [dict(id=row["id"], source=row["source"], series=row["series"],
                             observed=row["coordinates"]["current_a"]
                             if output == "current" else row["value"],
                             predicted=None, residual=None, status="rejected",
                             reason="No exact recorded temperature/gate slice")
                        for row in unmatched]
                diagnostics.append(dict(output=output, slice="unmatched",
                                        train=_metrics([]), holdout=_metrics(rows)))
    return diagnostics


def _plots(model, observations):
    plots = []
    by_id = {row["id"]: row for row in observations}
    for slice_ in model["slices"]:
        for output, poly in slice_["polynomials"].items():
            axis = ({"resistance": "t_j_c", "voltage": "current_a",
                     "current": "voltage_v"}.get(output, output))
            points = []
            selected = [by_id[row_id] for row_id in poly["training_ids"]]
            for row in observations:
                if row["split"] != "holdout":
                    continue
                if model["family"] == "diode" and any(
                        row["coordinates"][key] != value
                        for key, value in slice_["conditions"].items()):
                    continue
                if model["family"] == "energy" and any(
                        row["coordinates"][other] != model["anchor"][other]
                        for other in AXES if other != axis):
                    continue
                selected.append(row)
            for row in selected:
                x = row["value"] if output == "current" else row["coordinates"][axis]
                y = row["coordinates"]["current_a"] if output == "current" else row["value"]
                if poly["normalization"] is not None:
                    y /= poly["normalization"]
                if not np.isfinite(y):
                    raise ValueError("Normalized plot observation exceeds the numerical range")
                points.append(dict(id=row["id"], source=row["source"],
                                   series=row["series"], split=row["split"], x=x, y=y))
            lo, hi = poly["support"]
            samples = []
            for fraction in np.linspace(0, 1, 61):
                if fraction == 0:
                    coordinate = lo
                elif fraction == 1:
                    coordinate = hi
                elif np.isfinite(hi - lo):
                    # Offset interpolation retains adjacent floating-point
                    # bounds where separately weighted endpoints can round low.
                    coordinate = lo + (hi - lo) * fraction
                else:
                    coordinate = lo * (1 - fraction) + hi * fraction
                samples.append(dict(x=float(coordinate), y=_value(poly, coordinate)))
            unit = "factor" if poly["normalization"] is not None else (
                "J" if model["family"] == "energy" else UNITS[output])
            plots.append(dict(id=f"{slice_['id']}:{output}", output=output,
                              slice=slice_["id"], axis=axis,
                              x_label=f"{axis} [{AXIS_UNITS[axis]}]",
                              y_label=f"{output} [{unit}]", samples=samples,
                              observations=points, conditions=deepcopy(slice_["conditions"]),
                              normalization=poly["normalization"], support=list(poly["support"])))
    return plots


def train_device_model(request):
    """Fit supplied training observations and report train/holdout evidence."""
    _bounded(request)
    inputs = _inputs(request)
    model = _build(inputs)
    result = dict(inputs=inputs, model=model,
                  diagnostics=_diagnostics(model, inputs["observations"]),
                  plots=_plots(model, inputs["observations"]), scope=SCOPE,
                  units=deepcopy(UNITS), model_json=_canonical(model).decode("utf-8"))
    result["evidence"] = dict(method=METHOD, input_sha256=_hash(inputs),
                              dataset_sha256=_hash(inputs["observations"]),
                              model_sha256=_hash(model), result_sha256=_hash(result))
    return _bounded_result(result)


def evaluate_device_model(request):
    """Validate a JSON model and evaluate bounded, explicitly conditioned queries."""
    _bounded(request)
    _keys(request, ("model", "queries"), "Evaluation request")
    model = validate_device_model(request["model"])
    queries = request["queries"]
    if not isinstance(queries, list) or not 1 <= len(queries) <= MAX_ROWS:
        raise ValueError(f"Supply 1 to {MAX_ROWS} queries")
    ids, predictions = set(), []
    for query in queries:
        _keys(query, ("id", "output", "coordinates"), "Query")
        query_id = _text(query["id"], "Query ID")
        if query_id in ids:
            raise ValueError("Query IDs must be unique")
        ids.add(query_id)
        output = query["output"]
        if not isinstance(output, str) or output not in UNITS:
            raise ValueError("Unsupported model output")
        item = dict(id=query_id, output=output, status="rejected", value=None,
                    unit=UNITS[output], reason=None)
        try:
            item.update(value=_predict(model, output, query["coordinates"]),
                        status="predicted")
        except ValueError as exc:
            item["reason"] = str(exc)
        predictions.append(item)
    inputs = deepcopy(request)
    result = dict(predictions=predictions, scope=SCOPE)
    return _bounded_result(dict(inputs=inputs, result=result, evidence=dict(
        method=METHOD, input_sha256=_hash(inputs), model_sha256=_hash(model),
        result_sha256=_hash(result))))
