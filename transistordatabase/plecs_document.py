"""Nonexecuting PLECS table documents with explicit unsupported source content.

Source XML is retained independently of the supported scientific subset. Ratings
are deliberately absent: a table's coverage is not a device rating.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import math
import json
import base64
import re
from typing import Any
from xml.etree import ElementTree as ET

NAMESPACE = "http://www.plexim.com/xml/semiconductors/"
MAX_XML_BYTES = 8_000_000
MAX_NUMBERS = 100_000


@dataclass(frozen=True)
class Diagnostic:
    """A source location and the exact reason it cannot be evaluated."""

    code: str
    path: str
    message: str


@dataclass(frozen=True)
class Source:
    """Byte identity and encoding, distinct from submitted Unicode text identity."""

    name: str | None
    sha256: str
    encoding: str
    hash_scope: str


@dataclass(frozen=True)
class Slice:
    """One actual recorded SI curve prepared by Python for plotting."""

    label: str
    temperature_c: float
    supply_voltage_v: float | None
    unit: str
    plot_data: list[list[float]]


@dataclass(frozen=True)
class Table:
    """Raw scaled rectangular table plus supported, independently conditioned slices."""

    kind: str
    method: str
    status: str
    current_a: list[float]
    temperature_c: list[float]
    voltage_v: list[float] | None
    scale: float
    values: list
    gate_voltage_v: float | None
    gate_resistance_ohm: float | None
    formula: str | None
    source_path: str
    slices: list[Slice]
    attributes: dict[str, str]


@dataclass(frozen=True)
class Thermal:
    """Preserved thermal topology; Cauer elements never become Foster poles."""

    network_type: str
    status: str
    elements: list[dict[str, float]]
    source_path: str
    xml: str


@dataclass(frozen=True)
class Block:
    """One independent semiconductor block, without accidental component merging."""

    id: str
    device_type: str | None
    tables: list[Table]
    thermal: list[Thermal]
    preserved_metadata: list[dict[str, str]]


@dataclass(frozen=True)
class Package:
    """Source package identity with unknown ratings and preserved metadata."""

    id: str
    device_class: str | None
    vendor: str | None
    partnumber: str | None
    blocks: list[Block]
    thermal: list[Thermal]
    preserved_metadata: list[dict[str, str]]
    ratings: None = None
    fixed_conditions: dict | None = None


@dataclass(frozen=True)
class PlecsDocument:
    """A bounded nonexecuting source document and its supported table subset."""

    namespace_uri: str
    version: str | None
    source: Source
    source_xml: str
    source_base64: str
    status: str
    can_export: bool
    packages: list[Package]
    diagnostics: list[Diagnostic]
    preserved_metadata: list[dict[str, str]]
    schema_version: str = "plecs-document-1"
    can_archive: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return independent JSON-compatible data, including uninterpreted XML."""
        return asdict(self)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _namespace(tag: str) -> str:
    return tag[1:].split("}", 1)[0] if tag.startswith("{") else ""


def _children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in element if _local(child.tag) == name]


def _one(element: ET.Element, name: str) -> ET.Element | None:
    found = _children(element, name)
    if len(found) > 1:
        raise ValueError(f"Duplicate {name} elements")
    return found[0] if found else None


def _text(element: ET.Element, name: str) -> str | None:
    child = _one(element, name)
    return child.text.strip() if child is not None and child.text else None


def _number(value: str, path: str) -> float:
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{path}: expected numeric data") from exc
    if not math.isfinite(number):
        raise ValueError(f"{path}: nonfinite data is unsupported")
    return number


def _axis(element: ET.Element, name: str, path: str) -> list[float]:
    text = _text(element, name)
    if not text:
        raise ValueError(f"{path}/{name}: required explicit axis is missing")
    axis = [_number(token, f"{path}/{name}") for token in text.split()]
    if any(b <= a for a, b in zip(axis, axis[1:], strict=False)):
        raise ValueError(f"{path}/{name}: axis must be strictly increasing and unique")
    return axis


def _metadata(element: ET.Element) -> dict[str, str]:
    return {"kind": _local(element.tag), "xml": ET.tostring(element, encoding="unicode")}


def _attribute_number(elements, names, path):
    values = [_number(e.attrib[name], path) for e in elements for name in names
              if name in e.attrib]
    if values and any(value != values[0] for value in values):
        raise ValueError(f"{path}: conflicting condition attributes")
    return values[0] if values else None


def _table(element: ET.Element, path: str, diagnostics: list[Diagnostic]) -> Table:
    kind = {"ConductionLoss": "conduction", "TurnOnLoss": "turn-on",
            "TurnOffLoss": "turn-off"}[_local(element.tag)]
    method = _text(element, "ComputationMethod") or "Table only"
    formula = _text(element, "Formula")
    supported = method.lower() == "table only" and not formula
    if not supported:
        diagnostics.append(Diagnostic("unsupported-formula", path,
            "Formula computation is preserved as source and is never evaluated; "
            "base tables are not a complete model."))
    # Even formula placeholder tables are validated, but never plotted/evaluated.
    value_name = "VoltageDrop" if kind == "conduction" else "Energy"
    container = _one(element, value_name)
    if container is None:
        diagnostics.append(Diagnostic("missing-table", path, "No numeric table is available."))
        return Table(kind, method, "unsupported-formula" if not supported else "missing-data",
                     [], [], None, 1, [], None, None, formula, path, [], dict(element.attrib))
    current = _axis(element, "CurrentAxis", path)
    temperatures = _axis(element, "TemperatureAxis", path)
    voltage = None if kind == "conduction" else _axis(element, "VoltageAxis", path)
    scale = _number(container.get("scale", "1"), path + "/scale")
    if scale <= 0:
        raise ValueError(f"{path}: scale must be positive")
    expected = len(current) * len(temperatures) * (len(voltage) if voltage else 1)
    if expected > MAX_NUMBERS:
        raise ValueError(f"{path}: numeric table exceeds {MAX_NUMBERS} cells")
    temperature_rows = _children(container, "Temperature")
    if len(temperature_rows) != len(temperatures):
        raise ValueError(f"{path}: temperature row count differs from explicit axis")
    values, slices = [], []
    for t, row in zip(temperatures, temperature_rows, strict=True):
        rows = [row] if voltage is None else _children(row, "Voltage")
        if voltage is not None and len(rows) != len(voltage):
            raise ValueError(f"{path}: voltage row count differs from explicit axis")
        parsed_rows = []
        for index, value_row in enumerate(rows):
            numbers = [_number(token, path) for token in (value_row.text or "").split()]
            if len(numbers) != len(current):
                raise ValueError(f"{path}: row length differs from current axis")
            scaled = [value * scale for value in numbers]
            if any(not math.isfinite(value) for value in scaled):
                raise ValueError(f"{path}: scaled table values overflow")
            parsed_rows.append(numbers)
            if supported:
                supply = voltage[index] if voltage is not None else None
                label = f"{t:g} °C" + (f", {supply:g} V" if supply is not None else "")
                slices.append(Slice(label, t, supply, "V" if voltage is None else "J",
                                    [[i, y] for i, y in zip(current, scaled, strict=True)]))
        values.append(parsed_rows[0] if voltage is None else parsed_rows)
    gate = _attribute_number([element, container], ("Vg", "v_g", "GateVoltage"), path)
    resistance = _attribute_number([element, container], ("Rg", "rg", "GateResistance"), path)
    if resistance is not None and resistance < 0:
        raise ValueError(f"{path}: gate resistance must be nonnegative")
    known = {"ComputationMethod", "Formula", "CurrentAxis", "TemperatureAxis",
             "VoltageAxis", value_name}
    if any(_local(child.tag) not in known for child in element):
        diagnostics.append(Diagnostic("unsupported-table-metadata", path,
            "Unknown table content is preserved and requires explicit interpretation."))
    return Table(kind, method, "supported-table" if supported else "unsupported-formula",
                 current, temperatures, voltage, scale, values, gate, resistance,
                 formula, path, slices, dict(element.attrib))


def _thermal(element: ET.Element, path: str, diagnostics: list[Diagnostic]) -> list[Thermal]:
    networks = list(element) if _local(element.tag) == "ThermalModel" else [element]
    result = []
    for index, network in enumerate(networks):
        network_path = f"{path}/{_local(network.tag)}[{index}]"
        kind = network.get("type") or _local(network.tag)
        elements = []
        supported = kind == "Foster" and _local(network.tag) in ("Branch", "ThermalChain")
        for node in network:
            if _local(node.tag) in ("RTauElement", "RCElement"):
                attributes = {key: _number(value, network_path) for key, value in node.attrib.items()}
                required = {"R", "Tau"} if _local(node.tag) == "RTauElement" else {"R", "C"}
                if not required.issubset(attributes) or any(attributes[k] <= 0 for k in required):
                    raise ValueError(f"{network_path}: thermal elements require positive {required}")
                if set(attributes) != required or _local(node.tag) != "RTauElement":
                    supported = False
                elements.append(attributes)
            else:
                supported = False
        if not elements:
            supported = False
        if not supported:
            diagnostics.append(Diagnostic("unsupported-thermal", network_path,
                f"{kind} source retained; only explicit Foster R/Tau elements are evaluated. "
                "Cauer R/C elements are not Foster poles."))
        result.append(Thermal(kind, "supported-foster" if supported else "unsupported-content",
                              elements, network_path, ET.tostring(network, encoding="unicode")))
    return result


def parse_plecs_document(content: str | bytes, source_name: str | None = None) -> PlecsDocument:
    """Validate a bounded XML document without macros, formulas, includes or entities."""
    raw = content.encode("utf-8") if isinstance(content, str) else content
    if len(raw) > MAX_XML_BYTES:
        raise ValueError(f"PLECS source exceeds {MAX_XML_BYTES} bytes")
    declaration = re.search(br'encoding=[\"\']([^\"\']+)', raw[:200], re.I)
    encoding = declaration.group(1).decode("ascii") if declaration else "utf-8-sig"
    if b"\x00" in raw or (not isinstance(content, str) and encoding.lower().replace("_", "-") not in (
        "utf-8", "utf-8-sig", "iso-8859-1", "latin-1", "latin1", "windows-1252")):
        raise ValueError("Only UTF-8, Latin-1 and Windows-1252 XML byte encodings are supported")
    if re.search(br"<!\s*(?:DOCTYPE|ENTITY)", raw, re.IGNORECASE):
        raise ValueError("DTD and entity declarations are unsupported")
    try:
        root = ET.fromstring(content)
    except (ET.ParseError, ValueError) as exc:
        raise ValueError(f"Invalid PLECS XML: {exc}") from exc
    namespace = _namespace(root.tag)
    if _local(root.tag) != "SemiconductorLibrary" or namespace not in ("", NAMESPACE):
        raise ValueError("Expected SemiconductorLibrary in the PLECS namespace or no namespace")
    if any(_namespace(node.tag) != namespace for node in root.iter()):
        raise ValueError("Mixed or foreign XML namespaces are unsupported")
    nodes = list(root.iter())
    if len(nodes) > 100_000:
        raise ValueError("PLECS source exceeds 100000 XML nodes")
    pending = [(root, 1)]
    while pending:
        node, depth = pending.pop()
        if depth > 64:
            raise ValueError("PLECS source exceeds 64 XML nesting levels")
        pending.extend((child, depth + 1) for child in node)
    packages, diagnostics = [], []
    package_nodes = _children(root, "Package")
    if not package_nodes or len(package_nodes) > 128:
        raise ValueError("PLECS requires between 1 and 128 packages")
    # Preflight the complete derived numeric/plot workload before copying any
    # table into Python lists or SI chart pairs.
    planned_cells = 0
    for node in root.iter():
        name = _local(node.tag)
        if name in ("ConductionLoss", "TurnOnLoss", "TurnOffLoss") and (
            _one(node, "VoltageDrop" if name == "ConductionLoss" else "Energy") is not None
        ):
            current = _axis(node, "CurrentAxis", name)
            temperature = _axis(node, "TemperatureAxis", name)
            voltage_count = len(_axis(node, "VoltageAxis", name)) if name != "ConductionLoss" else 1
            planned_cells += len(current) * len(temperature) * voltage_count
            if planned_cells > MAX_NUMBERS:
                raise ValueError(f"PLECS document exceeds {MAX_NUMBERS} numeric/plot cells")
    for pindex, package in enumerate(package_nodes):
        path = f"Package[{pindex}]"
        blocks, thermal, metadata = [], [], []
        block_nodes = _children(package, "SemiconductorData")
        if len(block_nodes) > 512:
            raise ValueError("PLECS package exceeds 512 semiconductor blocks")
        for bindex, block in enumerate(block_nodes):
            block_path = f"{path}/SemiconductorData[{bindex}]"
            tables, block_thermal, block_metadata = [], [], []
            signatures = set()
            for child in block:
                name = _local(child.tag)
                if name in ("ConductionLoss", "TurnOnLoss", "TurnOffLoss"):
                    table = _table(child, f"{block_path}/{name}[{len(tables)}]", diagnostics)
                    for temperature in table.temperature_c:
                        for voltage in table.voltage_v or [None]:
                            signature = (name, child.get("gate"), table.gate_voltage_v,
                                         table.gate_resistance_ohm, temperature, voltage)
                            if signature in signatures:
                                raise ValueError(f"{block_path}: duplicate table conditions for {name}")
                            signatures.add(signature)
                    tables.append(table)
                elif name in ("ThermalChain", "ThermalModel"):
                    block_thermal.extend(_thermal(child, block_path, diagnostics))
                else:
                    block_metadata.append(_metadata(child))
                    diagnostics.append(Diagnostic("unsupported-block-content", block_path,
                                                  f"{name} is retained without interpretation."))
            blocks.append(Block(str(bindex), block.get("type"), tables,
                                block_thermal, block_metadata))
        for child in package:
            name = _local(child.tag)
            if name == "ThermalModel":
                thermal.extend(_thermal(child, path, diagnostics))
            elif name != "SemiconductorData":
                metadata.append(_metadata(child))
                if name not in ("Comment", "Variables", "Constants") or (
                    name in ("Variables", "Constants") and len(child)
                ):
                    diagnostics.append(Diagnostic("unsupported-package-content", path,
                        f"{name} is retained without evaluation or default substitution."))
        if not any(block.tables for block in blocks):
            diagnostics.append(Diagnostic("missing-table", path, "No semiconductor tables are available."))
        fixed_conditions = None
        for comment in _children(package, "Comment"):
            for line in _children(comment, "Line"):
                text = line.text or ""
                if text.startswith("TDB_FIXED_CONDITIONS_JSON="):
                    if fixed_conditions is not None:
                        raise ValueError("Duplicate explicit TDB fixed-condition metadata")
                    try:
                        fixed_conditions = json.loads(text.split("=", 1)[1],
                            parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite condition metadata")))
                    except (ValueError, TypeError, RecursionError) as exc:
                        raise ValueError("Malformed explicit TDB fixed-condition metadata") from exc
                    if not isinstance(fixed_conditions, dict):
                        raise ValueError("TDB fixed-condition metadata must be an object")
        packages.append(Package(str(pindex), package.get("class"), package.get("vendor"),
                                package.get("partnumber"), blocks, thermal, metadata,
                                fixed_conditions=fixed_conditions))
    metadata = [_metadata(child) for child in root if _local(child.tag) != "Package"]
    if metadata:
        diagnostics.append(Diagnostic("unsupported-library-content", "/", "Unknown library content is retained."))
    total_cells = sum(len(t.current_a) * len(t.temperature_c) * (len(t.voltage_v) if t.voltage_v else 1)
                      for p in packages for b in p.blocks for t in b.tables)
    if total_cells > MAX_NUMBERS:
        raise ValueError(f"PLECS document exceeds {MAX_NUMBERS} numeric cells")
    status = "supported-tables" if not diagnostics else "unsupported-content"
    if diagnostics and all(d.code == "missing-table" for d in diagnostics):
        status = "missing-data"
    # ElementTree honors the byte declaration. Retain exactly decoded source text
    # for archive export; UTF-8/Latin-1 are the supported upload encodings.
    if isinstance(content, str):
        source_xml = content
        encoding = "utf-8"
    else:
        try:
            source_xml = raw.decode(encoding)
        except (UnicodeError, LookupError) as exc:
            raise ValueError("Source encoding cannot be decoded safely") from exc
    return PlecsDocument(namespace, root.get("version"),
                         Source(source_name, sha256(raw).hexdigest(), encoding,
                                "submitted-utf8-text" if isinstance(content, str) else "original-bytes"),
                         source_xml, base64.b64encode(raw).decode("ascii"), status,
                         not diagnostics, packages, diagnostics, metadata)


def render_plecs_document(document: PlecsDocument, *, mode: str = "supported-tables") -> bytes:
    """Render verified tables or explicitly archive all preserved source content.

    Archive export is not a claim that formulas or a simulator model were validated.
    The source is reparsed to prevent forged intermediate objects bypassing validation.
    """
    raw = base64.b64decode(document.source_base64, validate=True)
    checked = parse_plecs_document(document.source_xml if document.source.hash_scope == "submitted-utf8-text" else raw,
                                   document.source.name)
    if mode == "source-preserved":
        declaration = re.search(r'encoding=[\"\']([^\"\']+)', checked.source_xml[:200], re.I)
        if document.source.hash_scope == "submitted-utf8-text" and declaration and (
            declaration.group(1).lower().replace("_", "-") not in ("utf-8", "utf-8-sig")
        ) and not checked.source_xml.isascii():
            raise ValueError("Source text contains non-ASCII characters and a non-UTF-8 declaration; "
                             "change the declaration explicitly to UTF-8, use supported-tables export, "
                             "or upload original file bytes for exact source preservation")
        return raw
    if mode != "supported-tables":
        raise ValueError("Select supported-tables or source-preserved export explicitly")
    if not checked.can_export:
        raise ValueError("Supported model export is unavailable: " + "; ".join(
            f"{d.path}: {d.message}" for d in checked.diagnostics))
    root = ET.fromstring(checked.source_xml)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def finalize_plecs_package(document: PlecsDocument, *, package_id: str,
                           ratings: dict, thermal_properties: dict,
                           block_targets: dict[str, str],
                           gate_voltage_by_table: dict[str, float] | None = None,
                           recovery_voltage_convention: str = "preserve-signed",
                           package_thermal_target: str | None = None):
    """Explicitly map one supported package into the existing required core model.

    Unknown ratings/areas cannot be inferred. Numeric gate declarations augment
    the mapping only; the original source and its unknowns remain in provenance.
    """
    import numpy as np
    from transistordatabase.core.models import (
        ChannelCharacteristics, ElectricalRatings, FosterThermalModel,
        SwitchingLossData, ThermalProperties, Transistor, TransistorMetadata,
    )
    if not document.can_export:
        raise ValueError("Core finalization requires a supported table-only document")
    package = next((p for p in document.packages if p.id == package_id), None)
    if package is None:
        raise ValueError("Unknown package_id")
    if not all((package.partnumber, package.vendor, package.device_class)):
        raise ValueError("Core finalization requires actual partnumber, vendor and device class")
    for key in ("v_abs_max", "i_abs_max", "i_cont", "t_j_max"):
        if key not in ratings or isinstance(ratings[key], bool) or not math.isfinite(ratings[key]):
            raise ValueError(f"Declare finite datasheet {key}")
    if any(ratings[k] <= 0 for k in ("v_abs_max", "i_abs_max", "i_cont")):
        raise ValueError("Electrical voltage/current ratings must be positive")
    if ratings["i_cont"] > ratings["i_abs_max"]:
        raise ValueError("Continuous current cannot exceed absolute current")
    for key in ("housing_area", "cooling_area"):
        if key not in thermal_properties or isinstance(thermal_properties[key], bool) or not math.isfinite(thermal_properties[key]) or thermal_properties[key] <= 0:
            raise ValueError(f"Declare positive {key} in m²")
    if set(block_targets) != {b.id for b in package.blocks}:
        raise ValueError("Map every semiconductor block explicitly")
    if len(set(block_targets.values())) != len(block_targets) or any(
        value not in ("switch", "diode") for value in block_targets.values()
    ):
        raise ValueError("Each block requires a distinct switch or diode target")
    if recovery_voltage_convention not in ("preserve-signed", "negative-to-positive-bus"):
        raise ValueError("Select an explicit supported recovery voltage convention")
    gates = gate_voltage_by_table or {}
    all_paths = {t.source_path for b in package.blocks for t in b.tables}
    if not set(gates).issubset(all_paths):
        raise ValueError("Gate declarations contain unknown table paths")
    if any(isinstance(v, bool) or not math.isfinite(v) for v in gates.values()):
        raise ValueError("Gate declarations must be finite numeric volts")
    declarations = dict(package_id=package_id, ratings=dict(ratings),
                        thermal_properties=dict(thermal_properties), block_targets=dict(block_targets),
                        gate_voltage_by_table=dict(gates),
                        recovery_voltage_convention=recovery_voltage_convention,
                        package_thermal_target=package_thermal_target)
    comment = json.dumps({"plecs_document_version": 1, "source_base64": document.source_base64,
                          "source": asdict(document.source), "declarations": declarations})
    transistor = Transistor(
        TransistorMetadata(package.partnumber, package.device_class, "PLECS explicit mapping",
                           package.vendor, "", comment=comment),
        ElectricalRatings(**ratings), ThermalProperties(**thermal_properties))

    def apply_thermal(networks, target):
        if not networks:
            return
        if len(networks) != 1 or networks[0].status != "supported-foster":
            raise ValueError("Core mapping requires one unambiguous Foster branch")
        elements = networks[0].elements
        if target.thermal_foster is not None:
            raise ValueError("Multiple thermal sources target the same component")
        total_resistance = sum(e["R"] for e in elements)
        if not math.isfinite(total_resistance):
            raise ValueError("Foster total resistance overflows finite core representation")
        target.thermal_foster = FosterThermalModel(
            r_th_vector=[e["R"] for e in elements], tau_vector=[e["Tau"] for e in elements],
            r_th_total=total_resistance)

    for block in package.blocks:
        target_name = block_targets[block.id]
        target = getattr(transistor, target_name)
        for table in block.tables:
            gate = table.gate_voltage_v
            declared = gates.get(table.source_path)
            if gate is not None and declared is not None and declared != gate:
                raise ValueError("Declared gate conflicts with an explicit source condition")
            gate = gate if gate is not None else declared
            if gate is None and (target_name == "switch" or table.kind != "conduction"):
                raise ValueError(f"{table.source_path}: declare the unknown gate condition explicitly")
            if table.kind == "conduction":
                for tindex, temperature in enumerate(table.temperature_c):
                    target.channel_data.append(ChannelCharacteristics(
                        temperature, np.array([[v * table.scale for v in table.values[tindex]],
                                               table.current_a]), gate))
            else:
                destination = (transistor.diode.e_rr_data if target_name == "diode" and table.kind == "turn-off"
                               else transistor.switch.e_on_data if target_name == "switch" and table.kind == "turn-on"
                               else transistor.switch.e_off_data if target_name == "switch" else None)
                if destination is None:
                    # PLECS diode turn-on is representable only as an explicit
                    # zero placeholder retained in source, never a switch curve.
                    if any(v != 0 for row in table.values for values in row for v in values):
                        raise ValueError("Nonzero diode turn-on has no core representation")
                    continue
                for tindex, temperature in enumerate(table.temperature_c):
                    for vindex, voltage in enumerate(table.voltage_v):
                        supply = voltage
                        if target_name == "diode" and recovery_voltage_convention == "negative-to-positive-bus":
                            if voltage > 0:
                                raise ValueError("Negative-to-positive recovery mapping requires nonpositive source voltages")
                            supply = -voltage
                        destination.append(SwitchingLossData(
                            "graph_i_e", temperature, supply, gate,
                            r_g=table.gate_resistance_ohm,
                            graph_i_e=np.array([table.current_a,
                                [v * table.scale for v in table.values[tindex][vindex]]]),
                            comment=json.dumps({"source_path": table.source_path,
                                "source_voltage_v": voltage,
                                "recovery_voltage_convention": recovery_voltage_convention})))
        apply_thermal(block.thermal, target)
    if package.thermal:
        if package_thermal_target is None:
            if len(block_targets) != 1:
                raise ValueError("Declare which component owns the package thermal branch")
            package_thermal_target = next(iter(block_targets.values()))
        if package_thermal_target not in block_targets.values():
            raise ValueError("Package thermal target must be a mapped component")
        apply_thermal(package.thermal, getattr(transistor, package_thermal_target))
    return transistor


def render_core_plecs(transistor) -> bytes:
    """Export unchanged source-backed core mappings without inventing conditions.

    Generic core curves need an explicit rectangular-table selection contract;
    this adapter fails clearly instead of silently picking the first gate slice.
    """
    try:
        metadata = json.loads(transistor.metadata.comment or "{}")
    except (TypeError, ValueError) as exc:
        raise ValueError("PLECS export requires an explicit source-backed mapping") from exc
    if (not isinstance(metadata, dict)
            or type(metadata.get("plecs_document_version")) is not int
            or metadata.get("plecs_document_version") != 1):
        raise ValueError("PLECS export requires a source-backed PLECS mapping; use the PLECS document workbench")
    source, declarations, encoded = (metadata.get(key) for key in ("source", "declarations", "source_base64"))
    required = {"package_id", "ratings", "thermal_properties", "block_targets"}
    optional = {"gate_voltage_by_table", "recovery_voltage_convention", "package_thermal_target"}
    if (not isinstance(source, dict)
            or not {"name", "sha256", "encoding", "hash_scope"} <= source.keys()
            or source["name"] is not None and not isinstance(source["name"], str)
            or any(not isinstance(source[key], str) for key in ("sha256", "encoding", "hash_scope"))
            or not isinstance(encoded, str) or not encoded or len(encoded) > 4 * ((MAX_XML_BYTES + 2) // 3)
            or not isinstance(declarations, dict) or not required <= declarations.keys()
            or set(declarations) - required - optional
            or not isinstance(declarations["package_id"], str)
            or any(not isinstance(declarations[key], dict) for key in ("ratings", "thermal_properties", "block_targets"))
            or declarations.get("gate_voltage_by_table") is not None
            and not isinstance(declarations["gate_voltage_by_table"], dict)):
        raise ValueError("Stored PLECS source evidence or declarations are malformed")
    try:
        source_bytes = base64.b64decode(encoded, validate=True)
        document = parse_plecs_document(
            source_bytes.decode("utf-8") if source["hash_scope"] == "submitted-utf8-text"
            else source_bytes, source["name"])
        expected = finalize_plecs_package(document, **declarations)
    except (TypeError, KeyError, UnicodeError) as exc:
        raise ValueError("Stored PLECS source evidence or declarations are malformed") from exc
    # Check the supported core subset against the source-derived mapping. All
    # conditions matter; a core edit must not export stale original XML.
    from dataclasses import fields
    import numpy as np

    def equal(a, b):
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and np.array_equal(a, b)
        if hasattr(a, "__dataclass_fields__") and hasattr(b, "__dataclass_fields__"):
            return all(equal(getattr(a, f.name), getattr(b, f.name)) for f in fields(a))
        if isinstance(a, list) and isinstance(b, list):
            return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b, strict=True))
        return a == b

    for attribute in ("metadata", "electrical_ratings", "thermal_properties"):
        if not equal(getattr(transistor, attribute), getattr(expected, attribute)):
            raise ValueError("Core metadata or declared conditions changed; finalize an updated source explicitly")
    for name in ("switch", "diode"):
        actual_component, expected_component = getattr(transistor, name), getattr(expected, name)
        for attribute in ("channel_data", "thermal_foster", "e_on_data", "e_off_data", "e_rr_data"):
            if hasattr(expected_component, attribute) and not equal(getattr(actual_component, attribute), getattr(expected_component, attribute)):
                raise ValueError("Core curves changed; source-backed PLECS export cannot emit stale source data")
    return render_plecs_document(document)
