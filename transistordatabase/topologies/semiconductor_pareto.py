"""Fixed-boundary semiconductor candidate screening."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from math import isfinite, prod
from numbers import Real
from typing import Any
from urllib.parse import urlsplit

from transistordatabase.core.models import Transistor
from transistordatabase.topologies.operating_point import (
    calculate_topology_operating_point,
)

_OBJECTIVE_DIRECTIONS = {
    "channel_conduction_loss_w": "minimize",
    "package_body_volume_mm3": "minimize",
    "component_cost_usd": "minimize",
}

_MODEL_LIMITATIONS = [
    "Loss includes switch and diode channel conduction only; switching, recovery, gate-drive, passive, and magnetic losses are excluded.",
    "The averaged CCM solve power-balances included channel losses and reports a separate inductor volt-second residual; nonlinear curves may not close both balances exactly.",
    "The rating screen checks only absolute blocking-voltage and peak-current ratings; it does not establish SOA, thermal safety, cooling, reliability, or margin.",
    "Parallel devices are treated as identical and share current equally; mismatch and layout effects are not modeled.",
    "Temperature is a one-pass fixed-temperature estimate and remains unavailable when required TDB thermal resistances are missing or nonpositive.",
    "Package volume is the body bounding-box product and excludes leads, board spacing, heatsinks, and assembly.",
]


def _positive_finite(value: Any) -> bool:
    """Return whether ``value`` is a finite positive real number.

    :param value: Value to validate.
    :type value: Any
    :returns: ``True`` for a finite real number greater than zero.
    :rtype: bool
    """
    numeric_value = _finite_real(value)
    return numeric_value is not None and numeric_value > 0


def _nonnegative_finite(value: Any) -> bool:
    """Return whether ``value`` is a finite nonnegative real number.

    :param value: Value to validate.
    :type value: Any
    :returns: ``True`` for a finite real number greater than or equal to zero.
    :rtype: bool
    """
    numeric_value = _finite_real(value)
    return numeric_value is not None and numeric_value >= 0


def _finite_real(value: Any) -> float | None:
    """Return a finite float for a real numeric input without raising.

    :param value: Value to normalize.
    :type value: Any
    :returns: Finite float, or ``None`` for non-real, non-finite, or overflowing
        input.
    :rtype: float | None
    """
    if not isinstance(value, Real) or isinstance(value, bool):
        return None
    try:
        numeric_value = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return numeric_value if isfinite(numeric_value) else None


def _valid_http_url(value: Any) -> bool:
    """Check that catalog provenance is an absolute HTTP or HTTPS URL.

    :param value: Catalog source value to validate.
    :type value: Any
    :returns: ``True`` when the value has an HTTP(S) scheme and a hostname.
    :rtype: bool
    """
    if not isinstance(value, str) or not value.strip():
        return False
    normalized = value.strip()
    if any(character.isspace() for character in normalized):
        return False
    try:
        parsed = urlsplit(normalized)
        port = parsed.port
        return (
            parsed.scheme.lower() in {"http", "https"}
            and bool(parsed.hostname)
            and (port is None or 1 <= port <= 65535)
        )
    except ValueError:
        return False


def _catalog_access_date(value: Any) -> str | None:
    """Normalize a valid catalog access date to ISO 8601 date form.

    :param value: Date object or ISO 8601 date string to validate.
    :type value: Any
    :returns: Normalized date, or ``None`` when the value is malformed.
    :rtype: str | None
    """
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        try:
            parsed = date.fromisoformat(value)
        except ValueError:
            return None
        if parsed.isoformat() == value:
            return parsed.isoformat()
    return None


def _catalog_item(
    part_id: str,
    catalog: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate and normalize one part's external catalog metadata.

    :param part_id: TDB identity used to associate the metadata with a part.
    :type part_id: str
    :param catalog: Request-scoped catalog entries keyed by TDB identity.
    :type catalog: Mapping[str, Any]
    :returns: Normalized item and no errors, or ``None`` and candidate-local errors.
    :rtype: tuple[dict[str, Any] | None, list[str]]
    """
    item = catalog.get(part_id)
    if not isinstance(item, Mapping):
        return None, [f"No sourced package and cost data supplied for '{part_id}'."]

    required = {
        "orderable_part": "an exact orderable part number",
        "unit_cost_usd": "a positive one-unit USD price",
        "length_mm": "a positive package-body length in mm",
        "width_mm": "a positive package-body width in mm",
        "height_mm": "a positive package-body height in mm",
        "price_source_url": "a price source URL",
        "cost_accessed_on": "a price access date",
        "dimensions_source_url": "a package-dimensions source URL",
    }
    errors: list[str] = []
    for key, description in required.items():
        value = item.get(key)
        if value is None or (isinstance(value, str) and not value.strip()):
            errors.append(f"Catalog data for '{part_id}' require {description}.")
    numeric_values: dict[str, float] = {}
    for key in ("unit_cost_usd", "length_mm", "width_mm", "height_mm"):
        value = _finite_real(item.get(key))
        if value is None or value <= 0:
            errors.append(
                f"Catalog '{key}' for '{part_id}' must be finite and greater than zero."
            )
        else:
            numeric_values[key] = value

    orderable_part = item.get("orderable_part")
    if orderable_part is not None and (
        not isinstance(orderable_part, str) or len(orderable_part.strip()) > 120
    ):
        errors.append(
            f"Catalog orderable_part for '{part_id}' must be a string of at most 120 characters."
        )

    for key in ("price_source_url", "dimensions_source_url"):
        value = item.get(key)
        if value is not None and value != "" and not _valid_http_url(value):
            errors.append(
                f"Catalog '{key}' for '{part_id}' must be an absolute HTTP(S) URL."
            )

    accessed_on = _catalog_access_date(item.get("cost_accessed_on"))
    if item.get("cost_accessed_on") is not None and accessed_on is None:
        errors.append(
            f"Catalog cost_accessed_on for '{part_id}' must be a valid ISO 8601 date."
        )

    if errors:
        return None, errors
    volume_mm3 = prod(numeric_values[key] for key in ("length_mm", "width_mm", "height_mm"))
    if not _positive_finite(volume_mm3):
        return None, [f"Package volume for '{part_id}' is not finite and positive."]

    return {
        "orderable_part": str(item["orderable_part"]).strip(),
        "unit_cost_usd": numeric_values["unit_cost_usd"],
        "length_mm": numeric_values["length_mm"],
        "width_mm": numeric_values["width_mm"],
        "height_mm": numeric_values["height_mm"],
        "price_source_url": str(item["price_source_url"]),
        "cost_accessed_on": accessed_on,
        "dimensions_source_url": str(item["dimensions_source_url"]),
        "volume_mm3": volume_mm3,
    }, []


def _rating_result(
    transistor: Transistor,
    *,
    role: str,
    blocking_voltage_v: float,
    peak_current_a: float,
    parallel_count: int,
) -> dict[str, Any]:
    """Compare one device branch's blocking/current demand with TDB ratings.

    :param transistor: TDB device selected for this switch or rectifier role.
    :type transistor: Transistor
    :param role: Candidate role used in the returned rating record.
    :type role: str
    :param blocking_voltage_v: Required absolute blocking voltage in volts.
    :type blocking_voltage_v: float
    :param peak_current_a: Branch peak current before parallel sharing, in amps.
    :type peak_current_a: float
    :param parallel_count: Number of identical parallel devices in the branch.
    :type parallel_count: int
    :returns: Individual voltage/current checks and their combined result.
    :rtype: dict[str, Any]
    """
    ratings = transistor.electrical_ratings
    voltage_limit = ratings.v_abs_max
    current_limit = ratings.i_abs_max
    voltage_pass = (
        _positive_finite(voltage_limit)
        and blocking_voltage_v <= float(voltage_limit)
    )
    current_per_device = peak_current_a / parallel_count
    current_pass = (
        _positive_finite(current_limit)
        and current_per_device <= float(current_limit)
    )
    return {
        "blocking_voltage_v": blocking_voltage_v,
        "voltage_rating_v": voltage_limit,
        "voltage_pass": bool(voltage_pass),
        "peak_current_per_device_a": current_per_device,
        "absolute_current_rating_a": current_limit,
        "current_pass": bool(current_pass),
        "pass": bool(voltage_pass and current_pass),
        "role": role,
    }


def calculate_boost_semiconductor_pareto(
    boundary: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    catalog: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate Boost CCM candidates and return their three-objective frontier.

    Cost and dimensions are request-supplied catalog metadata, separate from TDB
    records. Candidates without exact curve coverage, passing basic ratings, and
    complete sourced catalog metadata are returned as ineligible.

    :param boundary: Shared SI Boost CCM operating-point inputs.
    :type boundary: Mapping[str, Any]
    :param candidates: Candidate device pairs, counts, gate conditions, and
        resolved TDB objects. Each mapping receives an independent result.
    :type candidates: Sequence[Mapping[str, Any]]
    :param catalog: External package/price metadata keyed by TDB device identity.
    :type catalog: Mapping[str, Any]
    :returns: Boundary, objective rules, limitations, and per-candidate results.
    :rtype: dict[str, Any]
    :raises ValueError: If the candidate collection is empty.
    """
    if not candidates:
        raise ValueError("At least one semiconductor candidate is required.")

    parameters = {
        **dict(boundary),
        "mode": "loss-balanced",
        "conduction_regime": "CCM",
    }
    candidate_results: list[dict[str, Any]] = []

    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id", "")).strip()
        label = str(candidate.get("label", candidate_id)).strip()
        switch_id = str(candidate.get("switch_transistor_id") or "").strip()
        diode_id = str(candidate.get("diode_transistor_id") or "").strip()
        switch_device = candidate.get("switch_transistor")
        diode_device = candidate.get("diode_transistor")
        result: dict[str, Any] = {
            "candidate_id": candidate_id,
            "label": label,
            "switch_transistor_id": switch_id or None,
            "diode_transistor_id": diode_id or None,
            "parallel_switches": candidate.get("parallel_switches", 1),
            "parallel_diodes": candidate.get("parallel_diodes", 1),
            "eligible": False,
            "pareto": False,
            "ineligible_reasons": [],
            "operating_point": None,
            "ratings": None,
            "catalog": {},
            "objectives": None,
        }
        if candidate.get("device_load_error"):
            result["ineligible_reasons"].append(str(candidate["device_load_error"]))
        elif not isinstance(switch_device, Transistor):
            result["ineligible_reasons"].append(
                f"Switch transistor '{switch_id}' was not resolved from TDB."
            )
        elif not isinstance(diode_device, Transistor):
            result["ineligible_reasons"].append(
                f"Diode transistor '{diode_id}' was not resolved from TDB."
            )

        switch_catalog, switch_catalog_errors = _catalog_item(switch_id, catalog)
        diode_catalog, diode_catalog_errors = _catalog_item(diode_id, catalog)
        result["ineligible_reasons"].extend(switch_catalog_errors)
        result["ineligible_reasons"].extend(diode_catalog_errors)
        if switch_catalog is not None:
            result["catalog"]["switch"] = switch_catalog
        if diode_catalog is not None:
            result["catalog"]["diode"] = diode_catalog

        switch_count = candidate.get("parallel_switches", 1)
        diode_count = candidate.get("parallel_diodes", 1)
        if not isinstance(switch_count, int) or not 1 <= switch_count <= 100:
            result["ineligible_reasons"].append(
                "parallel_switches must be an integer from 1 to 100."
            )
        if not isinstance(diode_count, int) or not 1 <= diode_count <= 100:
            result["ineligible_reasons"].append(
                "parallel_diodes must be an integer from 1 to 100."
            )

        if isinstance(switch_device, Transistor) and isinstance(diode_device, Transistor):
            point_parameters = {
                **parameters,
                "parallel_switches": switch_count,
                "parallel_diodes": diode_count,
                "v_g": candidate.get("v_g"),
                "diode_v_g": candidate.get("diode_v_g"),
            }
            try:
                operating_point = calculate_topology_operating_point(
                    "boost",
                    point_parameters,
                    switch_device,
                    diode_device,
                    include_waveforms=False,
                )
                result["operating_point"] = operating_point
                calculated = operating_point["calculated"]
                ratings = {
                    "switch": _rating_result(
                        switch_device,
                        role="switch",
                        blocking_voltage_v=calculated["v_switch_block_v"],
                        peak_current_a=calculated["i_l_peak_a"],
                        parallel_count=switch_count,
                    ),
                    "diode": _rating_result(
                        diode_device,
                        role="diode",
                        blocking_voltage_v=calculated["v_diode_block_v"],
                        peak_current_a=calculated["i_diode_peak_a"],
                        parallel_count=diode_count,
                    ),
                }
                ratings["pass"] = ratings["switch"]["pass"] and ratings["diode"]["pass"]
                result["ratings"] = ratings
                for role in ("switch", "diode"):
                    rating = ratings[role]
                    if not rating["voltage_pass"]:
                        result["ineligible_reasons"].append(
                            f"{role.title()} absolute voltage rating does not cover the required blocking voltage."
                        )
                    if not rating["current_pass"]:
                        result["ineligible_reasons"].append(
                            f"{role.title()} absolute current rating does not cover the peak current per device."
                        )
            except (KeyError, TypeError, ValueError) as exc:
                result["ineligible_reasons"].append(str(exc))

        if (
            result["operating_point"] is not None
            and switch_catalog is not None
            and diode_catalog is not None
        ):
            losses = result["operating_point"]["losses"]
            objectives = {
                "channel_conduction_loss_w": losses["total_w"],
                "package_body_volume_mm3": (
                    switch_count * switch_catalog["volume_mm3"]
                    + diode_count * diode_catalog["volume_mm3"]
                ),
                "component_cost_usd": (
                    switch_count * switch_catalog["unit_cost_usd"]
                    + diode_count * diode_catalog["unit_cost_usd"]
                ),
            }
            if (
                _nonnegative_finite(objectives["channel_conduction_loss_w"])
                and _positive_finite(objectives["package_body_volume_mm3"])
                and _positive_finite(objectives["component_cost_usd"])
            ):
                result["objectives"] = objectives
            else:
                result["ineligible_reasons"].append(
                    "Pareto objectives must be finite; loss may be zero and volume/cost must be positive."
                )

        result["eligible"] = (
            not result["ineligible_reasons"]
            and result["operating_point"] is not None
            and result["ratings"] is not None
            and result["ratings"]["pass"]
            and result["objectives"] is not None
        )
        candidate_results.append(result)

    eligible = [candidate for candidate in candidate_results if candidate["eligible"]]
    for candidate in eligible:
        value = candidate["objectives"]
        dominated = any(
            all(
                other["objectives"][objective] <= value[objective]
                for objective in _OBJECTIVE_DIRECTIONS
            )
            and any(
                other["objectives"][objective] < value[objective]
                for objective in _OBJECTIVE_DIRECTIONS
            )
            for other in eligible
            if other is not candidate
        )
        candidate["pareto"] = not dominated

    return {
        "topology": "boost",
        "boundary": parameters,
        "mode": "loss-balanced",
        "conduction_regime": "CCM",
        "loss_scope": "channel_conduction_only",
        "objective_directions": dict(_OBJECTIVE_DIRECTIONS),
        "dominance_rule": (
            "All objectives are minimized. A candidate is dominated when another "
            "eligible candidate is no worse in all objectives and strictly better "
            "in at least one. Exact objective ties remain non-dominated."
        ),
        "cost_basis": "one-unit USD list price per physical package; counts are included",
        "eligible_candidate_count": len(eligible),
        "pareto_candidate_ids": [
            candidate["candidate_id"] for candidate in candidate_results if candidate["pareto"]
        ],
        "model_limitations": list(_MODEL_LIMITATIONS),
        "candidates": candidate_results,
    }
