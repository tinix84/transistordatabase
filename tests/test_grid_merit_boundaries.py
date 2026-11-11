"""Independent boundary checks for weighted merit scores and their evidence."""
import csv
import io
import json
from copy import deepcopy
from math import ulp

import pytest

from test_grid_merit import merit_fixture
from transistordatabase.grid_analysis import METRICS, evaluate_grid


def _comparison(values, objective_weights):
    """Construct one positive operating point and supply its accepted raw powers."""
    request = merit_fixture()
    config = request["configurations"][0]
    request["configurations"] = []
    for identity in values:
        entry = deepcopy(config)
        entry["id"] = entry["conduction"]["positive"]["source"] = identity
        request["configurations"].append(entry)
    request["operating_points"] = request["operating_points"][:1]
    request["objectives"] = [
        {"name": name, "unit": "W", "direction": "minimize", "weight": weight}
        for name, weight in objective_weights.items()
    ]

    def accepted_powers(data):
        conduction, on, off = values[data["conduction"]["positive"]["source"]]
        return {
            "total_power_w": conduction + on + off,
            "components": {name: {"power_w": value} for name, value in
                           (("conduction", conduction), ("turn_on", on), ("turn_off", off))},
        }

    return request, accepted_powers


@pytest.mark.parametrize(("middle", "high", "weights", "message"), [
    (ulp(0.0), 2, {"conduction_power_w": 1}, "penalty"),
    (0.25, 1, {"conduction_power_w": ulp(0.0), "turn_on_power_w": 1}, "contribution"),
])
def test_unrepresentable_positive_evidence_cannot_create_a_zero(middle, high, weights, message):
    """Detect a positive penalty or contribution that would round to zero."""
    request, evaluator = _comparison(
        {"low": (0, 0, 0), "middle": (middle, 1, 0), "high": (high, 1, 0)}, weights
    )
    with pytest.raises(ValueError, match=message):
        evaluate_grid(request, _evaluator=evaluator)


def test_unit_penalties_have_exact_unit_score_and_competition_rank():
    """Keep an exact boundary score when normalized weights sum above one in binary64."""
    request, evaluator = _comparison(
        {"B": (0, 0, 0), "C": (2, 2, 2), "A": (0, 0, 0)},
        {"conduction_power_w": 0.01, "turn_on_power_w": 0.01, "turn_off_power_w": 4},
    )
    merit = evaluate_grid(request, _evaluator=evaluator)["merit"]
    assert [(r["configuration_id"], r["score"], r["rank"]) for r in merit["ranking"]] == [
        ("A", 0, 1), ("B", 0, 1), ("C", 1, 3)
    ]


def test_rounded_coverage_cannot_admit_a_partial_metric():
    """Use point-level completeness even when retained and total weights round equal."""
    request, evaluator = _comparison(
        {"A": (10, 4, 0), "B": (20, 8, 0)}, {"conduction_power_w": 1, "turn_on_power_w": 1}
    )
    point = deepcopy(request["operating_points"][0])
    point.update(id="tiny", weight=1e-30)
    point["waveform"]["current_a"] = [3, 3]
    request["operating_points"].append(point)
    request["summary_mode"] = "successful-subset"

    def partially_available(data):
        powers = evaluator(data)
        if data["conduction"]["positive"]["source"] == "A" and data["waveform"]["current_a"][0] == 3:
            powers["components"]["turn_on"]["power_w"] = None
            powers["total_power_w"] = None
        return powers

    result = evaluate_grid(request, _evaluator=partially_available)
    merit = result["merit"]
    summary = result["summaries"][0]["metrics"]["turn_on_power_w"]
    assert (summary["selected_mean"], summary["coverage"], summary["complete"]) == (4, 1, False)
    assert merit["population"]["configuration_ids"] == ["B"]
    assert merit["objectives"][0]["minimum"] == 20
    assert merit["rows"][0]["exclusion_reasons"] == [
        {"name": "turn_on_power_w", "missing_operating_point_ids": ["tiny"]}
    ]
    excluded = [r for r in json.loads(merit["rows_json"]) if r["configuration_id"] == "A"]
    csv_excluded = [r for r in csv.DictReader(io.StringIO(merit["rows_csv"])) if r["configuration_id"] == "A"]
    assert len(excluded) == len(csv_excluded) == 2
    assert all(r["penalty"] is None and r["contribution"] is None for r in excluded)
    assert all(r["penalty"] == r["contribution"] == "NULL" for r in csv_excluded)


def test_full_audit_has_128_complete_unicode_rows():
    """Preserve all tied extrema, rows, and multibyte identities in both exports."""
    ids = [f'元件,"Δ"⚡{index:02d}' for index in range(32)]
    request, evaluator = _comparison({name: (1, 2, 3) for name in ids}, dict.fromkeys(METRICS, 1))
    merit = evaluate_grid(request, _evaluator=evaluator)["merit"]
    audit = json.loads(merit["rows_json"])
    csv_rows = list(csv.DictReader(io.StringIO(merit["rows_csv"])))
    assert audit == merit["export_rows"]
    assert len(audit) == len(csv_rows) == 128
    assert {(r["configuration_id"], r["name"]) for r in audit} == {
        (identity, name) for identity in ids for name in METRICS
    }
    for record, csv_record in zip(audit, csv_rows, strict=True):
        assert record["configuration_id"] == csv_record["configuration_id"]
        assert json.loads(csv_record["minimum_configuration_ids"]) == ids
        assert json.loads(csv_record["maximum_configuration_ids"]) == ids
        assert json.loads(csv_record["population_configuration_ids"]) == ids
