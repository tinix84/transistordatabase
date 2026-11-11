"""Explicit source-backed mapping from PLECS documents to core transistors.

Use parse_plecs_document for previews with unknown ratings. Core import requires
caller-provided datasheet ratings, areas, component mapping and unknown gate
conditions; table coverage never establishes absolute electrical ratings.
"""
from __future__ import annotations

from pathlib import Path

from transistordatabase.plecs_document import (
    finalize_plecs_package,
    parse_plecs_document,
)


def import_plecs_xml(file_path: str | Path, *, finalizations: list[dict] | None = None):
    """Map explicitly selected packages, preserving original source provenance.

    ``finalizations`` holds one finalize_plecs_package keyword mapping per package.
    Unknown document fields remain available through the separate preview API.
    """
    path = Path(file_path)
    document = parse_plecs_document(path.read_bytes(), str(path))
    if not finalizations:
        raise ValueError(
            "PLECS core import requires explicit finalizations (datasheet ratings, "
            "thermal areas, block targets and unknown gate conditions). Use "
            "parse_plecs_document or /api/plecs/preview to inspect source unknowns.")
    package_ids = [mapping.get("package_id") for mapping in finalizations]
    if len(set(package_ids)) != len(package_ids):
        raise ValueError("Duplicate package finalizations")
    return [finalize_plecs_package(document, **mapping) for mapping in finalizations]
