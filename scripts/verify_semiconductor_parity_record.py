"""Verify the identity and links in the November 2026 verification record.

Run with Python 3.10 or newer. This check only reads repository files.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "docs/reports/semiconductor-parity-verification-2026-11-10.json"
BASELINE = ROOT / "docs/sprints/semiconductor-function-gap-ledger.json"
BASELINE_SHA256 = "d744750e9bd140e0f6b61af3dcd7f8735ee954a5be799f73a8a7268a17651c71"


def check(condition: bool, message: str) -> None:
    """Stop at the first inconsistent record, including under optimized Python."""
    if not condition:
        raise ValueError(message)


def unique_keys(pairs: list[tuple]) -> dict:
    """Reject duplicate keys before the JSON decoder can discard their values."""
    obj = dict(pairs)
    check(len(pairs) == len(obj), "Duplicate JSON key")
    return obj


def main() -> None:
    """Check each baseline identity, November target, count and evidence file."""
    baseline_bytes = BASELINE.read_bytes()
    check(hashlib.sha256(baseline_bytes).hexdigest() == BASELINE_SHA256, "Baseline changed")
    historical = json.loads(baseline_bytes, object_pairs_hook=unique_keys)
    record = json.loads(RECORD.read_bytes(), object_pairs_hook=unique_keys)
    rows = record["definitions"]
    caps = record["capabilities"]
    files = record["files"]
    targets = record["targets"]
    counts = record["counts"]
    source_rows = {row["id"]: row for row in historical["definitions"]}
    check(record["baseline"]["sha256"] == BASELINE_SHA256, "Incorrect baseline reference")
    check(rows.keys() == source_rows.keys() and len(rows) == 3144, "Baseline ID mismatch")
    check(record["baseline"]["source_counts"] == Counter(row["source"] for row in rows.values()),
          "Source population mismatch")
    check(record["baseline"]["files"] == len(historical["files"]) == 505, "Baseline file count mismatch")
    check(counts["definitions"] == len(rows) and counts["capabilities"] == len(caps), "Totals differ")
    check(counts["source_function_equivalence_proofs"] == 0, "Unsupported equivalence credit")
    for identity, row in rows.items():
        original = source_rows[identity]
        for field in (
            "source", "path", "symbol", "kind", "line", "cell", "body_sha256",
            "definition_sha256", "source_file_sha256", "duplicate_of", "role",
        ):
            check(row[field] == original[field], f"{identity}: {field} differs")
        check(row["original_disposition"] == original["status"], f"{identity}: original disposition")
        check(bool(row["rationale"]), f"{identity}: missing rationale")
        check(set(row["capability_refs"]) <= caps.keys(), f"{identity}: unknown capability")
        permitted_targets = {target for cap in row["capability_refs"] for target in caps[cap]["targets"]}
        check(set(row["target_refs"]) <= permitted_targets, f"{identity}: unlinked specific target")
        if row["duplicate_of"]:
            check(row["duplicate_of"] in rows, f"{identity}: invalid duplicate")
        if original["status"] == "missing":
            check(row["capability_refs"] == ["digitizer"], f"{identity}: missing digitizer link")
    for key, field in (("original", "original_disposition"), ("classifications", "classification")):
        check(Counter(row[field] for row in rows.values()) == counts[key], f"{key}: counts differ")
    linked = [row for row in rows.values() if row["capability_refs"]]
    check(len(linked) == counts["records_with_capability_refs"], "Linked count differs")
    check(sum(r["source"] != "current" for r in linked) == counts["source_records_with_capability_refs"],
          "Historical source count differs")
    for source in historical["source_counts"]:
        expected = Counter(row["classification"] for row in rows.values() if row["source"] == source)
        check(expected == counts["classifications_by_source"][source], f"{source}: classifications differ")
    check(sum(bool(row["target_refs"]) for row in rows.values()) == counts["records_with_specific_target_refs"],
          "Specific target count differs")

    for path, evidence in files.items():
        item = (ROOT / path).resolve()
        check(item.is_relative_to(ROOT), f"External evidence path: {path}")
        data = item.read_bytes()
        check(len(data) == evidence["bytes"], f"{path}: size changed")
        check(hashlib.sha256(data).hexdigest() == evidence["sha256"], f"{path}: content changed")
    for key, cap in caps.items():
        check(bool(cap["scope"] and cap["limit"]), f"{key}: scope missing")
        check(bool(cap["targets"] and cap["python_tests"]), f"{key}: implementation evidence missing")
        check(set(cap["targets"]) <= targets.keys(), f"{key}: unknown target")
        for group in ("python_tests", "vue_tests", "views", "chromium_specs", "data"):
            check(set(cap.get(group, [])) <= files.keys(), f"{key}: missing {group} evidence")
    for key, target in targets.items():
        path = target["path"]
        check(path in files, f"{key}: missing file evidence")
        check(key == f"{path}#{target['symbol']}", f"{key}: target naming mismatch")
        raw = (ROOT / path).read_bytes()
        nodes = [n for n in ast.walk(ast.parse(raw)) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                 and n.name == target["symbol"].split(".")[-1] and n.lineno == target["line"]]
        check(len(nodes) == 1, f"{key}: definition moved")
        node = nodes[0]
        first = min([node.lineno] + [item.lineno for item in node.decorator_list])
        chunk = b"".join(raw.splitlines(keepends=True)[first - 1:node.end_lineno])
        check(node.end_lineno == target["end_line"] and hashlib.sha256(chunk).hexdigest()
              == target["definition_text_sha256"], f"{key}: definition changed")
    census = record["mat_source_census"]
    check(len(census["files"]) == census["file_count"] == 198, "Incorrect MAT file count")
    check(len({item["path"] for item in census["files"]}) == 198, "Duplicate MAT source")
    check(Counter(item["format"] for item in census["files"]) == census["formats"], "MAT formats differ")
    check(sum(item["byte_length"] for item in census["files"]) == census["byte_length"], "MAT sizes differ")
    manifest = hashlib.sha256()
    for item in sorted(census["files"], key=lambda entry: entry["path"]):
        manifest.update(f"{item['path']}\0{item['sha256']}\n".encode())
    check(manifest.hexdigest() == census["fingerprint"], "MAT source fingerprint differs")
    cohorts = [item for item in record["supplemental_sources"] if item["id"].startswith("catalog/")]
    check(len(cohorts) == 8 and sum(item["records"] for item in cohorts) == 2490,
          "Catalog population mismatch")
    hashed_sources = {item["source_sha256"] for item in historical["catalog_metric_census"]}
    hashed_sources.update(item["sha256"] for item in historical["files"] if item["source"] == "sc")
    for item in record["supplemental_sources"]:
        check(item["source_sha256"] in hashed_sources, f"{item['id']}: unknown source")
        check(set(item["capability_refs"]) <= caps.keys(), f"{item['id']}: unknown capability")
    for name in ("headless_python", "vue"):
        execution = record["final_verification"][name]
        population = execution["counts_by_file"]
        check(set(population) == set(execution["test_files"]), f"{name}: incomplete test population")
        check(set(population) <= files.keys(), f"{name}: missing file identity")
        check(len(population) == execution["files"] and sum(population.values()) == execution["passed"],
              f"{name}: execution counts differ")
        check(execution["failed"] == 0, f"{name}: execution had failures")
    report = RECORD.with_suffix(".md")
    for link in re.findall(r"\[[^\]]+\]\(([^()]+)\)", report.read_text(encoding="utf-8")):
        item = (report.parent / link.split("#", 1)[0]).resolve()
        check(item.is_relative_to(ROOT) and item.is_file(), f"Unresolved report link: {link}")
    print(f"Verified {len(rows)} baseline IDs, {len(caps)} capabilities, "
          f"{len(targets)} definitions and {len(files)} evidence files.")


if __name__ == "__main__":
    main()
