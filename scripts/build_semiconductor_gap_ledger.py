"""Build a static, reproducible function ledger without importing reference code.

The ledger deliberately separates inventory, candidate counterparts and verified
parity. Source-level similarity is never numerical or browser validation.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tokenize
from typing import Any


SC_ROOT = r"\\wsl.localhost\Ubuntu\home\tinix\claude_wsl\archive_ntbees2\sc"
ENCO_ROOT = r"D:\OneDrive\claude\bhy\enco"
IGNORED = {".git", "__pycache__", "node_modules", ".venv", "dist", "build"}
PLAN = "sprint-semiconductor-function-parity.md"
FAMILIES = {
    "gate-timing": (2, "Gate charge, drive and interval waveforms"),
    "dpt": (3, "Double pulse, raw/log extraction and energy maps"),
    "device-parameters": (4, "Device fitting, capacitances and signed slew"),
    "analytical": (4, "Analytical switching model"),
    "waveform-loss": (5, "Signed waveform conduction and event energies"),
    "regression-lut": (5, "Conditioned regression and lookup tables"),
    "time-series": (5, "Time and frequency series operations"),
    "parallel-thermal": (6, "Parallel/composite devices and thermal paths"),
    "converter": (7, "Converter branch loss aggregation"),
    "circuit-indicators": (7, "Circuit indicators and aggregation"),
    "catalog-fom": (8, "Catalog, FOM and Pareto exploration"),
    "plot": (8, "Linked plots, maps and surfaces"),
    "import-export": (8, "Import/export and database conversion"),
    "digitization": (8, "Image calibration and curve digitization"),
    "data-model": (8, "Device and condition data contracts"),
    "orchestration-context": (None, "Non-semiconductor orchestration/context"),
}


def sha(value: str | bytes) -> str:
    """Return a stable content digest."""
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def parser_runtime() -> dict[str, str]:
    """Record the interpreter needed to reproduce AST classifications."""
    return {
        "python_version": sys.version,
        "implementation": sys.implementation.name,
        "executable": sys.executable,
        "minimum_python": "3.12",
        "reason": "ENCO references use PEP 695 generic declarations; older Python parsers misclassify valid reference source as syntax errors.",
    }


def family(path: str, symbol: str, source: str) -> str:
    """Assign an execution family; broad context is explicitly outside port scope."""
    p, n = path.lower(), symbol.lower()
    key = p + " " + n
    if source == "enco" and any(x in p for x in (
        "/components/inductor/", "/components/capacitor/", "/components/resistor/",
        "transformer", "steinmetz", "/circuit/circuit/inductor", "/circuit/circuit/capacitor",
    )):
        return "orchestration-context"
    if source == "enco" and any(x in p for x in (
        "/_scripts/", "/_release/", "enco-cli/", "enco-documentation/",
        "enco_standards/", "enco_models/standards/", "enco_core/repository/",
        "enco_core/utility/version", "enco_core/log", "enco_core/path",
    )):
        return "orchestration-context"
    if "pareto" in key or "catalog" in key or "cost_index" in n or "volume_index" in n:
        return "catalog-fom"
    if any(x in key for x in ("datapicker", "analyze_image", "digitiz")):
        return "digitization"
    if any(x in key for x in ("thermal", "temperature.m", "temperature_delta", "parallel", "reverseturned", "derating")):
        return "parallel-thermal"
    if any(x in p for x in ("regressor", "/lut/")):
        return "regression-lut"
    if any(x in p for x in ("time_series", "frequency_series", "series_container")):
        return "time-series"
    if any(x in p for x in ("/dpt/", "double_pulse", "read_raw", "ltspice")):
        return "dpt"
    if any(x in key for x in ("biela", "transconductance", "reverse_recovery", "charge_stored", "charge_equivalent")):
        return "analytical"
    if any(x in key for x in ("dicdt", "dvcedt", "linear_model", "fit_diode", "capacit", "terminal_capacitor")):
        return "device-parameters"
    if any(x in key for x in ("gate_driver", "turnon_interval", "turnon_transition", "miller_plateau", "forcing_voltage", "calc_t1", "calc_t2", "calc_t3", "calc_tg", "gate_current", "exp_approx_vgs", "gatecharge")):
        return "gate-timing"
    if any(x in p for x in ("helpers_plot", "/plot/", "plot_functions")) or any(x in n for x in ("plot", "add_surface", "add_time_wfm", "viewer")):
        return "plot"
    if "circuit" in p and ("indicator" in n or "/circuit/circuit/" in p):
        return "circuit-indicators"
    if any(x in key for x in ("psemi", "semiconductor_loss", "dcdc_", "bridgeless", "pfc", "twolevel", "converter", "/models/dab", "/models/llc", "/models/cllc", "/psfb/")):
        return "converter"
    if any(x in key for x in ("pcond", "psw", "conduction", "conducting_power", "switching_loss", "switching_energy", "sw_losses", "switching_voltage_current", "transitions", "hardswitch", "rms")):
        return "waveform-loss"
    if any(x in key for x in ("xml", "plecs", "import", "export", "scdatabyname", "scdata_by_name", "convertsc", "save7", "xyzread", "xyz2grid", "/database/", "repository")):
        return "import-export"
    if source == "enco" and not any(x in key for x in ("mosfet", "diode", "indicator", "component", "circuit")):
        return "orchestration-context"
    return "data-model"


def py_signature(node: ast.AST, text: str) -> str:
    """Preserve the exact header through its top-level colon, including newlines."""
    fragment = "\n".join(text.splitlines()[node.lineno - 1:])
    depth = 0
    try:
        for token in tokenize.generate_tokens(io.StringIO(fragment).readline):
            if token.type != tokenize.OP:
                continue
            if token.string in "([{":
                depth += 1
            elif token.string in ")]}":
                depth -= 1
            elif token.string == ":" and depth == 0:
                lines = fragment.splitlines()
                return "\n".join(lines[:token.end[0] - 1] + [lines[token.end[0] - 1][:token.end[1]]])
    except tokenize.TokenError:
        pass
    return fragment.splitlines()[0]


def empty_statement(node: ast.stmt) -> bool:
    """Recognize pass, ellipsis and explicit NotImplemented placeholders."""
    return isinstance(node, ast.Pass) or (
        isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
        and node.value.value is Ellipsis
    ) or (isinstance(node, ast.Raise) and "NotImplemented" in ast.unparse(node))


def py_definitions(text: str, cell: int | None = None) -> tuple[list[dict], list[dict]]:
    """Extract functions and classes, including nested/local definitions."""
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        declarations, scopes = [], []
        lines = text.splitlines()
        for number, line in enumerate(lines, 1):
            match = re.match(r"^(\s*)(?:(async\s+)?def|class)\s+([A-Za-z_]\w*)\b", line)
            if not match:
                continue
            indent = len(match.group(1).expandtabs(8))
            while scopes and scopes[-1][0] >= indent:
                scopes.pop()
            name = ".".join([scope[1] for scope in scopes] + [match.group(3)])
            kind = "class" if line.lstrip().startswith("class ") else "function"
            header, end = line.strip(), number
            while not header.rstrip().endswith(":") and end < len(lines) and end < number + 30:
                header += "\n" + lines[end]
                end += 1
            declarations.append({
                "symbol": name, "kind": kind, "line": number, "end_line": None,
                "signature": header, "cell": cell, "stub": False,
                "body_sha256": None, "definition_sha256": None, "decorators": [],
                "extraction_method": "declaration-only-regex-in-syntax-error-source",
                "body_status": "unknown-syntax-error", "parse_error": f"{exc.msg}, line {exc.lineno}",
            })
            scopes.append((indent, match.group(3)))
        return declarations, [{"line": exc.lineno, "message": exc.msg, "cell": cell}]
    rows = []

    def visit(node: ast.AST, prefix: str = "") -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                body = list(child.body)
                if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                    body = body[1:]
                kind = "class" if isinstance(child, ast.ClassDef) else "function"
                row = {
                    "symbol": prefix + child.name, "kind": kind,
                    "line": child.lineno, "end_line": child.end_lineno,
                    "signature": py_signature(child, text), "cell": cell,
                    "stub": kind == "function" and (not body or all(empty_statement(x) for x in body)),
                    "body_sha256": sha(ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False)),
                    "definition_sha256": sha(ast.dump(child, include_attributes=False)),
                    "decorators": [ast.unparse(x) for x in child.decorator_list],
                }
                rows.append(row)
                visit(child, prefix + child.name + ".")
            else:
                visit(child, prefix)

    visit(tree)
    return rows, []


def matlab_code_lines(text: str) -> list[str]:
    """Strip MATLAB comments while respecting quoted strings and block comments."""
    result, block = [], False
    for raw in text.splitlines():
        if raw.strip().startswith("%{"):
            block = True
        if block:
            result.append("")
            if raw.strip().startswith("%}"):
                block = False
            continue
        quote, out, i = None, [], 0
        while i < len(raw):
            char = raw[i]
            if quote:
                out.append(char)
                if char == quote:
                    if i + 1 < len(raw) and raw[i + 1] == quote:
                        out.append(raw[i + 1])
                        i += 1
                    else:
                        quote = None
            elif char == "%":
                break
            elif char == '"' or (char == "'" and (not out or out[-1] in " =([,{;")):
                quote = char
                out.append(char)
            else:
                out.append(char)
            i += 1
        result.append("".join(out))
    return result


def matlab_definitions(text: str) -> list[dict]:
    """Read multiline MATLAB/Octave headers and inline empty function bodies.

    Segment boundaries are the next function declaration, not an inferred MATLAB
    block end. The field body_extent records this conservative hashing boundary.
    """
    lines = matlab_code_lines(text)
    starts = [i for i, line in enumerate(lines) if re.match(r"^\s*function\b", line)]
    rows = []
    for index, start in enumerate(starts):
        stop = starts[index + 1] if index + 1 < len(starts) else len(lines)
        header, end = lines[start].strip(), start
        while header.rstrip().endswith("...") and end + 1 < stop:
            end += 1
            header = header.rstrip()[:-3] + " " + lines[end].strip()
        declaration = re.sub(r"^function\s+", "", header).split(";", 1)[0].strip()
        assignment = re.match(r"^(?:\[[^]]*\]|[A-Za-z]\w*)\s*=\s*(.*)", declaration)
        rhs = assignment.group(1).strip() if assignment else declaration
        match = re.match(r"([A-Za-z]\w*(?:\.[A-Za-z]\w*)*)", rhs)
        if not match:
            raise ValueError(f"Unrecognized MATLAB function header at {start + 1}: {header}")
        body = "\n".join(lines[end + 1:stop]).strip()
        inline_tail = header.split(";", 1)[1].strip() if ";" in header else ""
        semantic_body = re.sub(r"\b(end|endfunction)\b[;\s]*", "", inline_tail + "\n" + body).strip()
        rows.append({
            "symbol": match.group(1), "kind": "function", "line": start + 1,
            "end_line": stop, "signature": "\n".join(text.splitlines()[start:end + 1]).strip(), "cell": None,
            "stub": not semantic_body,
            "body_sha256": sha(body), "definition_sha256": sha(header + "\n" + body),
            "body_extent": "through next function declaration or EOF; nested end not inferred",
            "decorators": [],
        })
    return rows


def example_data(text: str, path: Path, root: Path, file_index: set[str]) -> list[dict]:
    """Record literal data references; unresolved relative paths are not proven absent."""
    result = []
    suffixes = r"(?:mat|csv|xlsx|xlsm|xml|raw|log|txt|json|asc|cir|net)"
    for match in re.finditer(r"['\"]([^'\"\n]+\." + suffixes + r")[\"']", text, re.I):
        value = match.group(1)
        if len(value) > 250 or any(char in value for char in "{}<>"):
            continue
        candidate = Path(value)
        candidates = [candidate] if candidate.is_absolute() else [path.parent / candidate, root / candidate]
        found = [str(p) for p in candidates if os.path.normcase(os.path.abspath(p)) in file_index]
        result.append({
            "literal": value, "line": text[:match.start()].count("\n") + 1,
            "status": "present" if found else "unresolved-example-data",
            "resolved_paths": found,
            "note": "Checked source parent and source root; caller working directory/dynamic composition may differ.",
        })
    return result


def scan(root: Path, source: str) -> tuple[list[dict], list[dict]]:
    """Inventory every Python/MATLAB/notebook file under a source tree read-only."""
    rows, files = [], []
    if not root.is_dir():
        raise FileNotFoundError(f"Source root missing or inaccessible: {root}")
    paths = []
    for directory, directories, names in os.walk(root):
        directories[:] = [name for name in directories if name not in IGNORED]
        paths.extend(Path(directory) / name for name in names)
    paths.sort()
    file_index = {os.path.normcase(os.path.abspath(p)) for p in paths}
    for path in paths:
        if path.suffix.lower() not in {".py", ".m", ".ipynb"} or any(x in IGNORED for x in path.parts):
            continue
        raw = path.read_bytes()
        text = raw.decode("utf-8-sig", errors="replace")
        relative = path.relative_to(root).as_posix()
        record = {
            "source": source, "source_root": str(root), "path": relative,
            "sha256": sha(raw), "lines": len(text.splitlines()), "parse_errors": [],
            "example_data": example_data(text, path, root, file_index),
        }
        definitions = []
        if path.suffix.lower() == ".py":
            definitions, record["parse_errors"] = py_definitions(text)
        elif path.suffix.lower() == ".m":
            definitions = matlab_definitions(text)
        else:
            notebook = json.loads(text)
            record["cells"] = []
            for cell_index, cell in enumerate(notebook.get("cells", [])):
                content = "".join(cell.get("source", []))
                record["cells"].append({"index": cell_index, "type": cell["cell_type"], "lines": len(content.splitlines()), "output_count": len(cell.get("outputs", []))})
                if cell["cell_type"] == "code":
                    cell_rows, errors = py_definitions(content, cell_index)
                    definitions.extend(cell_rows)
                    record["parse_errors"].extend(errors)
        for row in definitions:
            row.update({"source": source, "source_root": str(root), "path": relative,
                        "source_file_sha256": record["sha256"]})
            identity = f"{source}:{relative}:{row['cell']}:{row['symbol']}:{row['kind']}"
            # Distinguish property setter/getter and repeated local definitions.
            occurrence = sum(r["source"] == source and r["path"] == relative and r["cell"] == row["cell"] and r["symbol"] == row["symbol"] and r["kind"] == row["kind"] for r in rows)
            row["id"] = source + "-" + sha(identity + ":" + str(occurrence))[:16]
            row["occurrence"] = occurrence
            row["family"] = family(relative, row["symbol"], source)
            row["sequence_step"] = FAMILIES[row["family"]][0]
            row["role"] = "test-reference" if any(x.startswith("test") for x in Path(relative).parts) or row["symbol"].split(".")[-1].startswith("test_") else ("notebook-reference" if path.suffix == ".ipynb" else "source")
            if source == "current" and (relative.startswith("gui/") or relative.startswith("frontend/")):
                row["role"] = "legacy-qt-reference"
            rows.append(row)
        record["definition_count"] = len(definitions)
        files.append(record)
    return rows, files


TARGETS = {
    "gate-timing": [("analytical_models.py", "GateChargeModel.calc_turn_on_times")],
    "dpt": [("utils/ltspice_dpt.py", "LTspiceDPT.analyze_waveforms")],
    "device-parameters": [("analytical_models.py", "calc_device_capacitances"), ("core/models.py", "Switch.linearize_at_operating_point")],
    "analytical": [("analytical_models.py", "ChristenBielaModel.calc_switching_energy")],
    "waveform-loss": [("waveform_losses.py", "calc_total_loss")],
    "regression-lut": [("topologies/operating_point.py", "_switching_curve")],
    "time-series": [("waveform_losses.py", "WaveformData")],
    "parallel-thermal": [("core/models.py", "FosterThermalModel.get_thermal_impedance")],
    "converter": [("topologies/operating_point.py", "calculate_topology_operating_point")],
    "circuit-indicators": [("topologies/operating_point.py", "calculate_topology_operating_point")],
    "catalog-fom": [("catalog_fom.py", "screen_catalog")],
    "plot": [("plot_functions.py", "plot_soa_lib")],
    "import-export": [("plecs_importer.py", "import_plecs_xml")],
    "data-model": [("core/models.py", "Transistor")],
}


def annotate(rows: list[dict]) -> None:
    """Attach conservative counterpart candidates and content-based duplicate links."""
    current = {(r["path"], r["symbol"]): r for r in rows if r["source"] == "current"}
    seen = {}
    for row in rows:
        key = (row["source"], row["kind"], row["definition_sha256"])
        row["duplicate_of"] = seen.get(key) if row["definition_sha256"] else None
        if row["definition_sha256"]:
            seen.setdefault(key, row["id"])
        row["current_targets"] = []
        for suffix, symbol in TARGETS.get(row["family"], []):
            target = current.get((suffix, symbol))
            if target:
                row["current_targets"].append({"id": target["id"], "path": target["path"], "line": target["line"], "symbol": target["symbol"], "evidence": "Static candidate in the same capability family; no function-level parity proven."})
        row["evidence"] = []
        row["validation"] = {"numerical": "not-run", "api": "not-run", "browser": "not-run"}
        row["gap"] = "Compare domains, units, conditions and failure semantics against the target; add independent numerical/API/browser evidence."
        if row["stub"]:
            status = "stub"
            row["gap"] = "Source placeholder; body does not establish a previously working capability."
        elif row["duplicate_of"]:
            status = "duplicate"
            row["gap"] = "Identical normalized definition hash; retain source identity and assess canonical capability once."
        elif row["role"] in {"test-reference", "notebook-reference", "legacy-qt-reference"} or row["family"] == "orchestration-context":
            status = "reference-only"
            row["gap"] = "Reference/test/example or unrelated orchestration; do not transplant this as a missing product function."
        elif row["source"] == "current":
            status = "Python-only"
            row["gap"] = "Current definition inventoried; this ledger has no function-level browser or numerical parity proof."
        elif row["current_targets"]:
            status = "candidate-match-needs-validation"
        else:
            status = "missing"
            row["gap"] = "No reviewed current counterpart selected; this is an inventory gap, not proof no possible implementation exists."
        row["status"] = status
        if row.get("body_status") == "unknown-syntax-error":
            row["status"] = "reference-only"
            row["gap"] = "Declaration present in syntactically invalid source; body and executable behavior unknown. Reimplement intended science from validated equations."
            row["evidence"].append(row["parse_error"])
        # Source defects confirmed by actual body/caller inspection.
        if row["source"] == "enco" and "/components/mosfet/" in row["path"]:
            if row["symbol"] in {"MOSFET.get_vf", "MOSFET.get_switching_energy"}:
                row["status"] = "broken-adapter" if not row["duplicate_of"] else "duplicate"
                row["evidence"].append("Public wrapper passes numpy ndarray; diode feature map / EnergyRegressor.predict() uses Polars DataFrame.select(). R_on uses a separate sklearn ndarray pipeline and is not implicated. Low-level tests cannot validate this wrapper contract.")
            if row["symbol"] == "enco_to_xml":
                row["status"] = "stub"
                row["gap"] = "Exporter writes fixed example MOSFET Specs (100 V/50 A); it does not serialize supplied device curves."
                row["evidence"].append("Body inspected: fixed example XML fields and no data-set serialization.")
        if row["source"] == "enco" and row["symbol"] == "ParetoPlot":
            row["status"] = "reference-only" if not row["duplicate_of"] else "duplicate"
            row["evidence"].append("Schema only. Inspected generate_plots dispatcher handles TimeSeries/FrequencySeries/VBarGrouped/VBarStacked; ParetoPlot reaches Unknown plot type.")
        if row["source"] == "sc" and row["path"] == "semiconductors/mosfet.py":
            row["evidence"].append("Archive body contains undefined-variable/inconsistent interval expressions; waveform intent requires independent fixture validation (review: sc-archive-parity-review.md).")
        if row["source"] == "current" and row["path"] == "gui_web/backend/main.py" and row["symbol"] in {"calculate_biela", "generate_dpt_netlist"}:
            row["status"] = "broken-adapter"
            row["evidence"].append("Initial phase-0 inspection found obsolete analytical model names or LTspiceDPT constructor/file-return mismatch. Recheck after steps 3/4.")


def reconcile(rows: list[dict], files: list[dict], old_inventory: dict) -> dict:
    """Independently reconcile every old definition and directly recheck source lines."""
    sc = [r for r in rows if r["source"] == "sc" and r["kind"] == "function" and r["cell"] is None]
    found = Counter((r["path"], r["line"]) for r in sc)
    old = Counter((f["path"], d["line"]) for f in old_inventory["files"] for d in f.get("definitions", []))
    missing = list((old - found).elements())
    extras = list((found - old).elements())
    checked = 0
    source_cache = {}
    for row in rows:
        path = Path(row["source_root"]) / row["path"]
        if path not in source_cache:
            source_cache[path] = path.read_text(encoding="utf-8-sig", errors="replace")
        text = source_cache[path]
        if row["cell"] is not None:
            text = "".join(json.loads(text)["cells"][row["cell"]]["source"])
        line = text.splitlines()[row["line"] - 1].strip()
        expected = r"function\b" if path.suffix.lower() == ".m" else r"(?:async\s+def|def|class)\b"
        if not re.match(expected, line):
            raise AssertionError(f"Source location mismatch: {row['id']} {line}")
        checked += 1
    if missing:
        raise AssertionError(f"Original archive definitions omitted: {missing}")
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), "Stable IDs must be unique"
    return {"old_archive_definitions": sum(old.values()), "accounted_old_definitions": sum(old.values()) - len(missing), "missing_old_definitions": missing, "additional_non_notebook_archive_functions": extras, "source_locations_verified": checked, "unique_ids": len(set(ids)), "files": len(files), "method": "Old inventory path/line multiset reconciliation plus direct source-line checks, separate from extraction."}


def report(ledger: dict[str, Any]) -> str:
    """Render actual individual function lists grouped by source and capability."""
    rows, files = ledger["definitions"], ledger["files"]
    out = ["# Semiconductor function gap analysis", "", "Phase 1 of [the execution plan](sprint-semiconductor-function-parity.md#1-complete-the-function-ledger). Static discovery baseline; no new parity is claimed.", "", "Reference roots (read-only):", "", f"- `{ledger['roots']['sc']}`", f"- `{ledger['roots']['enco']}`", "", "The JSON ledger retains every function/class, source root, exact header, line, notebook cell, body/definition/file hash, duplicate identity, role, candidate target, gap, status and sequence step. Classes are included so schema-only features cannot masquerade as functions. Notebook line numbers are cell-local and always include a cell index. MATLAB body hashes conservatively stop at the next declaration; Python hashes use normalized ASTs. Duplication requires identical definition content, never a matching name alone.", "", "## Scope and status meanings", "", "- `missing`: no reviewed current counterpart selected; investigate before implementation.", "- `candidate-match-needs-validation`: related current code exists; units, conditions, regime and behavior remain unproven.", "- `Python-only`: current source exists; this baseline has no function-level numerical/API/browser proof.", "- `broken-adapter`: source inspection finds a caller/callee contract mismatch.", "- `stub`, `duplicate`, `reference-only`: preserve evidence without counting each as an independent missing capability.", "- `browser-verified`: reserved for function-specific numerical, API and browser evidence; no baseline row receives it.", "", "General ENCO CLI, repositories, documentation, optimizers, magnetic/passive design and simulation orchestration are comparison context, not missing semiconductor features to transplant. Source tests remain test references. Existing Qt code remains a screenshot/reference source and is outside migration targets.", "", "## Family overview", "", "| Family | SC definitions | ENCO definitions | Current definitions | Execution step |", "| --- | ---: | ---: | ---: | --- |"]
    runtime = ledger["parser_runtime"]
    scope_index = out.index("## Scope and status meanings")
    out[scope_index:scope_index] = [
        "## Parser runtime and reproduction", "",
        f"Python **{runtime['python_version'].split()[0]}** ({runtime['implementation']}); interpreter `{runtime['executable']}`.", "",
        "Use Python **3.12 or newer** for this inventory generator: ENCO uses PEP 695 generic declarations. An older parser can report valid reference files as syntax errors and change body hashes, duplicate links and classifications.", "",
        "The current-package index is a **pre-implementation baseline**. Scientific modules may change during the subsequent numbered implementation steps; their execution evidence belongs in a separate reviewed overlay. The baseline does not assert that stored current-source hashes describe those later edits.", "",
        "Reproduce from this repository in PowerShell (regeneration refreshes the current index):", "",
        "```powershell",
        f"& '{runtime['executable']}' scripts/build_semiconductor_gap_ledger.py",
        "```", "",
    ]
    for fam, (step, title) in FAMILIES.items():
        counts = Counter(r["source"] for r in rows if r["family"] == fam)
        out.append(f"| {title} | {counts['sc']} | {counts['enco']} | {counts['current']} | {step or 'context only'} |")
    out += ["", "Counts include classes, notebook functions and duplicate copies; they are not unique working product features.", "", "## ENCO findings and implementation boundaries", "", "- The ENCO root has several generations: `enco/packages`, standalone `enco-core`, `enco-analysis`, `enco-circuits`, `enco-library`, `enco-database`, and `library-design`. All source sets are inventoried, including the project circuit examples. Identical copies are linked by definition hash; different bodies remain distinct.", "- MOSFET regressors expose conduction resistance, forward diode voltage/current, switching energies, transitions and waveform loss indices. Separable LUTs are condition-normalized estimates; their training and error metrics do not establish arbitrary multidimensional empirical coverage. Current adapters must retain measured conditions and explicit missing/estimate scope.", "- Inspected scalar `MOSFET.get_vf/get_switching_energy` wrappers pass NumPy arrays to regressors whose `predict` methods select Polars columns. These wrappers are broken-adapter evidence even where low-level regressor tests exist.", "- PLECS XML import extracts substantive conduction and switching datasets. `enco_to_xml` writes fixed example XML and is a placeholder, so bidirectional round-trip parity is unproven.", "- `ParetoPlot` is a schema. The inspected plot dispatcher lacks a Pareto branch. Its linked time/frequency plots and grouped/stacked indicator bars are substantive visual references; do not credit a supported Pareto renderer.", "- TimeSeries/SeriesContainer supply arithmetic, RMS, interpolation, FFT and transition operations. Preserve explicit sample alignment, signed current, event conventions, exact conditions and distinct zero/missing states when implementing the lean shared Python engine.", "- Circuit indicator attachment aggregates RMS, maximum voltage, cost/volume and loss indices, transition statistics and circuit/group totals. Indices and fixed-temperature loss sums are not a validated electrothermal solver. Future iteration must expose convergence status.", "- ENCO copyright headers reserve rights to BRUSA. Inspect locally for behavior; write independent implementations here and do not publish reference bodies.", "", "## Validation and inventory reconciliation", "", "```json", json.dumps(ledger["verification"], indent=2), "```", "", "The original archive multiset is reconciled in full. Additional notebook definitions/classes are supplemental. Source syntax errors and unresolved literal example data below are retained, never imported or executed. A literal may require a caller working directory or dynamic path, so unresolved references are not all proven missing files.", "", "## Syntax errors", ""]
    for f in files:
        for error in f["parse_errors"]:
            cell = f" cell {error['cell']}" if error.get("cell") is not None else ""
            out.append(f"- `{f['source']}:{f['path']}:{error['line']}{cell}` — {error['message']}")
    out += ["", "## Workbook and catalog metric capabilities", "", "The five user-supplied catalog files are represented by eight imported source sheets / CSVs (2,490 rows) in the existing portable snapshot. This function census does not execute spreadsheet formulas or macros. A full census of formulas on unimported/summary sheets and VBA procedures remains **open**. Repeated formula cells do not count as distinct functions.", "", "`SemiconductorsData.xlsm` contains temperature-dependent resistance estimates, Coss×Rds and Qg×Rds FOMs, maximum loss at 100 °C, a derived DC current, current/cost, voltage×current/cost, voltage/total-switching-energy and its cost ratio. These require independent unit/condition review: its MOSFET FOM references the later 150 °C resistance column while the current browser's primary Rds uses the first explicit 25 °C condition. The workbook DC current also uses typical resistance while another FOM uses maximum resistance. Preserve these distinctions before enabling workbook-equivalent comparisons.", "", "Each following representative formula family is **reference-needs-validation**. Formula text and source-cell references are provenance, not validated thermal/current predictions. Existing catalog FOM numerical/browser evidence is documented separately in `sc-archive-parity-review.md`.", ""]
    for source in ledger["catalog_metric_census"]:
        out.append(f"- `{source['file']}` / `{source['sheet']}`: {source['records']} imported records; " + ("; ".join(f"`{m['header']}` (`{m['cell']}`, `{m['formula']}`)" for m in source["formula_families"]) or "No formula families in the imported row snapshot; source catalog ratings and cost remain condition-specific."))
    out += ["", "## Unresolved example-data references", ""]
    for f in files:
        unresolved = [r for r in f["example_data"] if r["status"] != "present"]
        if unresolved:
            out.append(f"- `{f['source']}:{f['path']}`: " + "; ".join(f"`{r['literal']}` (line {r['line']})" for r in unresolved))
    out += ["", "## Individual reference functions by implementation family", "", "Every SC/ENCO definition is listed below. Repeated files are retained so source coverage is auditable. Headers and body hashes are in the JSON. Candidate targets identify areas to investigate, never asserted parity."]
    for fam, (step, title) in FAMILIES.items():
        out += ["", f"### {title}", "", f"Execution: step {step} of [{PLAN}]({PLAN})" if step else "Execution: context only; no automatic port.", ""]
        groups = defaultdict(list)
        for row in rows:
            if row["source"] != "current" and row["family"] == fam:
                groups[(row["source"], row["path"])].append(row)
        for (source, path), group in sorted(groups.items()):
            out += [f"**{source}: `{path}`**", ""]
            for r in group:
                loc = str(r["line"]) + (f" / cell {r['cell']}" if r["cell"] is not None else "")
                target = ", ".join(f"`{t['path']}:{t['line']} {t['symbol']}`" for t in r["current_targets"])
                dup = f"; canonical `{r['duplicate_of']}`" if r["duplicate_of"] else ""
                out.append(f"- `{r['symbol']}` ({r['kind']}, line {loc}) — **{r['status']}**{dup}" + (f"; candidate {target}" if target and r["status"] not in {"duplicate", "reference-only", "stub"} else "") + f". ID `{r['id']}`.")
            out.append("")
    out += ["## Current package source index", "", "Current definitions remain in the JSON individually. This grouped index lists actual names for navigation; no browser parity is inferred.", ""]
    groups = defaultdict(list)
    for r in rows:
        if r["source"] == "current":
            groups[r["path"]].append(r)
    for path, group in sorted(groups.items()):
        out.append(f"- `{path}`: " + "; ".join(f"`{r['symbol']}`:{r['line']} ({r['status']})" for r in group))
    out += ["", "## Completion rule", "", "Update a row only with an exact current callable, independent numerical fixture, API equality evidence where exposed, and exercised browser behavior where required by the plan. Re-run this generator for source inventories; preserve subsequent execution evidence in a separately reviewed overlay before regeneration. This baseline does not overwrite the existing FOM report or count its prior aggregate test totals as proof for unrelated functions.", ""]
    return "\n".join(out)


def catalog_metric_census(repository: Path) -> list[dict]:
    """Summarize preserved workbook formula families without opening/altering sources."""
    snapshot = json.loads((repository / "transistordatabase/data/catalogs/archive_sc.json").read_text(encoding="utf-8"))
    result = []
    for source in snapshot["sources"]:
        families = {}
        for record in snapshot["records"]:
            if record["source_id"] != source["id"]:
                continue
            for column, formula in record.get("formulas", {}).items():
                header = record.get("raw", {}).get(column, {}).get("header", column)
                if header in families:
                    continue
                number, letters = int(column), ""
                while number:
                    number, remainder = divmod(number - 1, 26)
                    letters = chr(65 + remainder) + letters
                families[header] = {"header": header, "cell": f"{source['sheet']}!{letters}{record['row']}", "formula": formula, "status": "reference-needs-validation", "sequence_step": 8}
        result.append({"file": source["file"], "sheet": source["sheet"], "source_sha256": source["sha256"], "records": source["record_count"], "formula_families": list(families.values()), "scope": "Imported rows only; summary sheets/VBA census open."})
    return result


def main() -> None:
    """Write only the two requested reports and print compact verification results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sc-root", type=Path, default=Path(SC_ROOT))
    parser.add_argument("--enco-root", type=Path, default=Path(ENCO_ROOT))
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    if sys.version_info < (3, 12):
        parser.error("Use Python 3.12+ to parse ENCO PEP 695 declarations correctly.")
    rows, files = [], []
    for source, root in (("sc", args.sc_root), ("enco", args.enco_root), ("current", args.repository / "transistordatabase")):
        extracted, scanned = scan(root, source)
        rows.extend(extracted)
        files.extend(scanned)
        print(f"Scanned {source}: {len(scanned)} files, {len(extracted)} definitions", flush=True)
    annotate(rows)
    previous = json.loads((args.repository / "docs/sprints/sc-archive-function-inventory.json").read_text(encoding="utf-8"))
    verification = reconcile(rows, files, previous)
    ledger = {
        "schema_version": 1, "baseline_date": "2026-10-15", "plan": PLAN,
        "parser_runtime": parser_runtime(),
        "current_index_scope": "Pre-implementation baseline; later implementation evidence belongs in a separate reviewed overlay.",
        "roots": {"sc": str(args.sc_root), "enco": str(args.enco_root), "current": str(args.repository / "transistordatabase")},
        "method": "Read-only AST/MATLAB/notebook structural analysis; no source execution, no name-only parity, no reference body copies.",
        "verification": verification, "status_counts": dict(Counter(r["status"] for r in rows)),
        "source_counts": dict(Counter(r["source"] for r in rows)),
        "files": files, "definitions": rows,
        "catalog_metric_census": catalog_metric_census(args.repository),
    }
    destination = args.repository / "docs/sprints"
    (destination / "semiconductor-function-gap-ledger.json").write_text(json.dumps(ledger, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (destination / "semiconductor-function-gap-analysis.md").write_text(report(ledger), encoding="utf-8")
    print(json.dumps({"verification": verification, "source_counts": ledger["source_counts"], "status_counts": ledger["status_counts"]}, indent=2))


if __name__ == "__main__":
    main()
