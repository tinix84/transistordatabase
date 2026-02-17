"""
FastAPI backend for Transistor Database Web Application.

Uses the real core services and JsonTransistorRepository for persistent
storage instead of in-memory placeholder dicts.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from transistordatabase.backend.concrete_services import (
    ConcreteServiceFactory,
    ExportService,
    PlottingService,
    ValidationService,
    ComparisonService,
)
from transistordatabase.core.models import (
    Transistor,
    TransistorMetadata,
    ElectricalRatings,
    ThermalProperties,
    ChannelCharacteristics,
    SwitchingLossData,
    GateChargeCurve,
    SOA,
    VoltageDependentCapacitance,
)
from transistordatabase.core.pair_repository import JsonPairRepository
from transistordatabase.core.repository import (
    JsonTransistorRepository,
    JsonTransistorLoader,
)
from transistordatabase.gui_web.backend.schemas import (
    ChannelCurveCreate,
    SwitchingLossCreate,
    GateChargeCurveCreate,
    SOACreate,
    CapacitanceCurveCreate,
    CurveValidationResult,
    PLECSExportOptions,
    MATLABExportOptions,
    BatchExportRequest,
    TopologyCalculationRequest,
    AdvancedSearchRequest,
    UserSettings,
    BielaModelRequest,
    GateChargeModelRequest,
    DPTConfigRequest,
    PLECSImportResponse,
)
from transistordatabase.gui_web.backend.topology_service import TopologyCalculatorService

logger = logging.getLogger(__name__)

# In-memory settings storage (use database in production)
_user_settings: Dict[str, UserSettings] = {}

app = FastAPI(
    title="Transistor Database API",
    description="REST API for managing transistor data",
    version="1.0.0",
)

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:5174", "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Service and repository wiring
# ---------------------------------------------------------------------------

# Default JSON database directory (sibling of the package root)
_default_db_dir = Path(__file__).resolve().parent.parent.parent.parent / "database"
_repo = JsonTransistorRepository(_default_db_dir)

# Default switching pairs directory
_default_pairs_dir = Path(__file__).resolve().parent.parent.parent.parent / "switching_pairs"
_pair_repo = JsonPairRepository(_default_pairs_dir)

_services = ConcreteServiceFactory.create_all_services()
_validation: ValidationService = _services['validation']
_comparison: ComparisonService = _services['comparison']
_export: ExportService = _services['export']
_plotting: PlottingService = _services['plotting']
_loader = JsonTransistorLoader()
_topology_calc = TopologyCalculatorService()


def _get_transistor(name: str) -> Transistor:
    """Get transistor by name, raise 404 if not found."""
    t = _repo.get_by_name(name)
    if t is None:
        raise HTTPException(status_code=404, detail=f"Transistor '{name}' not found")
    return t


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def transistor_to_dict(transistor: Transistor) -> Dict[str, Any]:
    """Convert Transistor object to dictionary for API responses."""
    result: Dict[str, Any] = {
        "metadata": {
            "name": transistor.metadata.name,
            "type": transistor.metadata.type,
            "manufacturer": transistor.metadata.manufacturer,
            "housing_type": transistor.metadata.housing_type,
            "author": transistor.metadata.author,
            "comment": transistor.metadata.comment,
        },
        "electrical": {
            "v_abs_max": transistor.electrical_ratings.v_abs_max,
            "i_abs_max": transistor.electrical_ratings.i_abs_max,
            "i_cont": transistor.electrical_ratings.i_cont,
            "t_j_max": transistor.electrical_ratings.t_j_max,
        },
        "thermal": {
            "r_th_cs": transistor.thermal_properties.r_th_cs,
            "housing_area": transistor.thermal_properties.housing_area,
            "cooling_area": transistor.thermal_properties.cooling_area,
        },
        "switch": {
            "channel_count": len(transistor.switch.channel_data),
            "e_on_count": len(transistor.switch.e_on_data),
            "e_off_count": len(transistor.switch.e_off_data),
            "has_foster": transistor.switch.thermal_foster is not None,
            "gate_charge_count": len(transistor.switch.gate_charge_curves),
            "soa_count": len(transistor.switch.soa),
        },
        "diode": {
            "channel_count": len(transistor.diode.channel_data),
            "e_rr_count": len(transistor.diode.e_rr_data),
            "has_foster": transistor.diode.thermal_foster is not None,
        },
        "capacitances": {
            "c_oss_count": len(transistor.c_oss),
            "c_iss_count": len(transistor.c_iss),
            "c_rss_count": len(transistor.c_rss),
        },
    }
    return result


def dict_to_transistor(data: Dict[str, Any]) -> Transistor:
    """Convert dictionary to Transistor object.

    Handles both nested format (from web interface) and flat format (TDB files).
    """
    if "metadata" in data and "electrical" in data and "thermal" in data:
        metadata = TransistorMetadata(
            name=data["metadata"]["name"],
            type=data["metadata"]["type"],
            manufacturer=data["metadata"]["manufacturer"],
            housing_type=data["metadata"]["housing_type"],
            author=data["metadata"].get("author", ""),
            comment=data["metadata"].get("comment", ""),
        )
        electrical = ElectricalRatings(
            v_abs_max=data["electrical"]["v_abs_max"],
            i_abs_max=data["electrical"]["i_abs_max"],
            i_cont=data["electrical"]["i_cont"],
            t_j_max=data["electrical"]["t_j_max"],
        )
        thermal = ThermalProperties(
            r_th_cs=data["thermal"]["r_th_cs"],
            r_th_switch_cs=data["thermal"].get("r_th_switch_cs"),
            r_th_diode_cs=data["thermal"].get("r_th_diode_cs"),
            housing_area=data["thermal"]["housing_area"],
            cooling_area=data["thermal"]["cooling_area"],
        )
    else:
        metadata = TransistorMetadata(
            name=data.get("name", "Unknown"),
            type=data.get("type", "Unknown"),
            manufacturer=data.get("manufacturer", "Unknown"),
            housing_type=data.get("housing_type", "Unknown"),
            author=data.get("author", ""),
            comment=data.get("comment", ""),
        )
        t_j_max = data.get("t_j_max", data.get("t_c_max", 150))
        if t_j_max is None:
            t_j_max = 150
        electrical = ElectricalRatings(
            v_abs_max=data.get("v_abs_max", 0),
            i_abs_max=data.get("i_abs_max", 0),
            i_cont=data.get("i_cont", 0),
            t_j_max=t_j_max,
        )
        thermal = ThermalProperties(
            r_th_cs=data.get("r_th_cs", 0),
            r_th_switch_cs=data.get("r_th_switch_cs"),
            r_th_diode_cs=data.get("r_th_diode_cs"),
            housing_area=data.get("housing_area", 0),
            cooling_area=data.get("cooling_area", 0),
        )

    return Transistor(metadata=metadata, electrical=electrical, thermal=thermal)


# ---------------------------------------------------------------------------
# Core CRUD endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Transistor Database API", "version": "1.0.0"}


@app.get("/api/transistors")
async def get_transistors() -> List[Dict[str, Any]]:
    """Get all transistors from the repository."""
    names = _repo.list_all()
    result = []
    for name in names:
        t = _repo.get_by_name(name)
        if t is not None:
            result.append(transistor_to_dict(t))
    return result


@app.get("/api/transistors/{transistor_id}")
async def get_transistor(transistor_id: str) -> Dict[str, Any]:
    """Get a specific transistor by name."""
    return transistor_to_dict(_get_transistor(transistor_id))


@app.post("/api/transistors")
async def create_transistor(transistor_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new transistor."""
    try:
        transistor = dict_to_transistor(transistor_data)
        _repo.save(transistor)
        return {"id": transistor.metadata.name, "message": "Transistor created successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid transistor data: {e!s}")


@app.put("/api/transistors/{transistor_id}")
async def update_transistor(transistor_id: str, transistor_data: Dict[str, Any]) -> Dict[str, Any]:
    """Update an existing transistor."""
    _get_transistor(transistor_id)  # Verify it exists
    try:
        transistor = dict_to_transistor(transistor_data)
        _repo.save(transistor)
        return {"message": "Transistor updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid transistor data: {e!s}")


@app.delete("/api/transistors/{transistor_id}")
async def delete_transistor(transistor_id: str) -> Dict[str, Any]:
    """Delete a transistor."""
    if not _repo.delete(transistor_id):
        raise HTTPException(status_code=404, detail="Transistor not found")
    return {"message": "Transistor deleted successfully"}


# ---------------------------------------------------------------------------
# Validation & Comparison
# ---------------------------------------------------------------------------

@app.post("/api/transistors/{transistor_id}/validate")
async def validate_transistor(transistor_id: str) -> Dict[str, Any]:
    """Validate a transistor."""
    transistor = _get_transistor(transistor_id)
    return _validation.validate_transistor(transistor)


@app.post("/api/transistors/compare")
async def compare_transistors(transistor_ids: List[str]) -> Dict[str, Any]:
    """Compare multiple transistors (basic comparison)."""
    if len(transistor_ids) < 2:
        raise HTTPException(status_code=400, detail="At least 2 transistors required")
    transistors = [_get_transistor(tid) for tid in transistor_ids]
    return _comparison.compare_transistors(transistors)


@app.post("/api/comparison/advanced")
async def advanced_comparison(request: Dict[str, Any]) -> Dict[str, Any]:
    """Advanced comparison with 9 plot types and configuration.

    Request body:
    {
        "transistor_ids": ["id1", "id2", "id3"],
        "config": {
            "t_j": 25,
            "v_supply": [600, 600, 600],
            "r_g_on": [10, 15, 20],
            "r_g_off": [10, 15, 20],
            "parallel_count": [1, 2, 1],
            "i_channel": 50
        }
    }
    """
    transistor_ids = request.get('transistor_ids', [])
    config = request.get('config', {})

    if not 2 <= len(transistor_ids) <= 3:
        raise HTTPException(
            status_code=400,
            detail="Advanced comparison requires 2-3 transistors"
        )

    transistors = [_get_transistor(tid) for tid in transistor_ids]
    return _comparison.generate_advanced_comparison(transistors, config)


# ---------------------------------------------------------------------------
# Export endpoints
# ---------------------------------------------------------------------------

@app.post("/api/transistors/{transistor_id}/export/{format}")
async def export_transistor(transistor_id: str, format: str) -> FileResponse:
    """Export a transistor in the specified format.

    Supported formats: json, csv, spice, plecs, matlab, gecko, ltspice.
    """
    transistor = _get_transistor(transistor_id)
    supported = ("json", "csv", "spice", "plecs", "matlab", "gecko", "ltspice")
    if format not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{format}'. Supported: {', '.join(supported)}",
        )

    tmp_dir = Path(tempfile.mkdtemp(prefix="tdb_export_"))
    try:
        if format == "json":
            out = tmp_dir / f"{transistor_id}.json"
            _export.export_to_json(transistor, out)
            return FileResponse(out, media_type="application/json", filename=out.name)

        if format == "csv":
            out = tmp_dir / f"{transistor_id}.csv"
            _export.export_to_csv([transistor], out)
            return FileResponse(out, media_type="text/csv", filename=out.name)

        if format == "spice":
            out = tmp_dir / f"{transistor_id}.spice"
            _export.export_to_spice(transistor, out)
            return FileResponse(out, media_type="text/plain", filename=out.name)

        if format == "plecs":
            _export.export_to_plecs(transistor, tmp_dir)
            xml_files = list(tmp_dir.glob("*.xml"))
            if not xml_files:
                raise HTTPException(status_code=500, detail="PLECS export produced no files")
            return FileResponse(xml_files[0], media_type="application/xml", filename=xml_files[0].name)

        if format == "matlab":
            _export.export_to_matlab(transistor, tmp_dir)
            mat_files = list(tmp_dir.glob("*.mat"))
            if not mat_files:
                raise HTTPException(status_code=500, detail="MATLAB export produced no files")
            return FileResponse(mat_files[0], media_type="application/octet-stream", filename=mat_files[0].name)

        if format == "gecko":
            _export.export_to_gecko_circuits(transistor, {"output_path": str(tmp_dir)})
            scl_files = list(tmp_dir.glob("*.scl"))
            if not scl_files:
                raise HTTPException(status_code=500, detail="GeckoCIRCUITS export produced no files")
            return FileResponse(scl_files[0], media_type="text/plain", filename=scl_files[0].name)

        if format == "ltspice":
            out = tmp_dir / f"{transistor_id}.asc"
            _export.export_to_ltspice(transistor, out)
            return FileResponse(out, media_type="text/plain", filename=out.name)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {e!s}")


@app.get("/api/transistors/{transistor_id}/export/{format}/preview")
async def export_preview(transistor_id: str, format: str) -> Dict[str, Any]:
    """Preview export content (first 100 lines for text formats).

    Returns preview text and metadata about the export.
    """
    transistor = _get_transistor(transistor_id)
    supported = ("json", "csv", "spice", "plecs", "ltspice")
    if format not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Preview not supported for '{format}'. Supported: {', '.join(supported)}",
        )

    tmp_dir = Path(tempfile.mkdtemp(prefix="tdb_preview_"))
    try:
        preview_lines = []
        file_size = 0
        file_name = ""

        if format == "json":
            out = tmp_dir / f"{transistor_id}.json"
            _export.export_to_json(transistor, out)
            content = out.read_text()
            lines = content.split('\n')
            preview_lines = lines[:100]
            file_size = len(content)
            file_name = out.name

        elif format == "csv":
            out = tmp_dir / f"{transistor_id}.csv"
            _export.export_to_csv([transistor], out)
            content = out.read_text()
            lines = content.split('\n')
            preview_lines = lines[:100]
            file_size = len(content)
            file_name = out.name

        elif format == "spice":
            out = tmp_dir / f"{transistor_id}.spice"
            _export.export_to_spice(transistor, out)
            content = out.read_text()
            lines = content.split('\n')
            preview_lines = lines[:100]
            file_size = len(content)
            file_name = out.name

        elif format == "plecs":
            _export.export_to_plecs(transistor, tmp_dir)
            xml_files = list(tmp_dir.glob("*.xml"))
            if not xml_files:
                raise HTTPException(status_code=500, detail="PLECS export produced no files")
            content = xml_files[0].read_text()
            lines = content.split('\n')
            preview_lines = lines[:100]
            file_size = len(content)
            file_name = xml_files[0].name

        elif format == "ltspice":
            out = tmp_dir / f"{transistor_id}.asc"
            _export.export_to_ltspice(transistor, out)
            content = out.read_text()
            lines = content.split('\n')
            preview_lines = lines[:100]
            file_size = len(content)
            file_name = out.name

        return {
            "preview": '\n'.join(preview_lines),
            "total_lines": len(lines) if 'lines' in locals() else 0,
            "preview_lines": len(preview_lines),
            "file_size": file_size,
            "file_name": file_name,
            "truncated": len(lines) > 100 if 'lines' in locals() else False,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preview failed: {e!s}")


@app.post("/api/transistors/batch_export")
async def batch_export(request: BatchExportRequest, background_tasks: BackgroundTasks) -> StreamingResponse:
    """Export multiple transistors in a ZIP archive.

    Request body:
    {
        "transistor_ids": ["id1", "id2", "id3"],
        "format": "json",
        "options": {}
    }

    Returns a ZIP file containing all exports.
    """
    if not request.transistor_ids:
        raise HTTPException(status_code=400, detail="No transistor IDs provided")

    if request.format not in ("json", "csv", "spice", "plecs", "matlab", "gecko", "ltspice"):
        raise HTTPException(status_code=400, detail=f"Unsupported format: {request.format}")

    # Create ZIP in memory
    zip_buffer = io.BytesIO()

    try:
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            tmp_dir = Path(tempfile.mkdtemp(prefix="tdb_batch_"))

            for transistor_id in request.transistor_ids:
                try:
                    transistor = _get_transistor(transistor_id)
                    export_dir = tmp_dir / transistor_id

                    if request.format == "json":
                        out = export_dir / f"{transistor_id}.json"
                        out.parent.mkdir(parents=True, exist_ok=True)
                        _export.export_to_json(transistor, out)
                        zip_file.write(out, arcname=f"{transistor_id}.json")

                    elif request.format == "csv":
                        out = export_dir / f"{transistor_id}.csv"
                        out.parent.mkdir(parents=True, exist_ok=True)
                        _export.export_to_csv([transistor], out)
                        zip_file.write(out, arcname=f"{transistor_id}.csv")

                    elif request.format == "spice":
                        out = export_dir / f"{transistor_id}.spice"
                        out.parent.mkdir(parents=True, exist_ok=True)
                        _export.export_to_spice(transistor, out)
                        zip_file.write(out, arcname=f"{transistor_id}.spice")

                    elif request.format == "plecs":
                        export_dir.mkdir(parents=True, exist_ok=True)
                        _export.export_to_plecs(transistor, export_dir)
                        for xml_file in export_dir.glob("*.xml"):
                            zip_file.write(xml_file, arcname=f"{transistor_id}/{xml_file.name}")

                    elif request.format == "matlab":
                        export_dir.mkdir(parents=True, exist_ok=True)
                        _export.export_to_matlab(transistor, export_dir)
                        for mat_file in export_dir.glob("*.mat"):
                            zip_file.write(mat_file, arcname=f"{transistor_id}/{mat_file.name}")

                    elif request.format == "gecko":
                        export_dir.mkdir(parents=True, exist_ok=True)
                        _export.export_to_gecko_circuits(transistor, {"output_path": str(export_dir)})
                        for scl_file in export_dir.glob("*.scl"):
                            zip_file.write(scl_file, arcname=f"{transistor_id}/{scl_file.name}")

                    elif request.format == "ltspice":
                        out = export_dir / f"{transistor_id}.asc"
                        out.parent.mkdir(parents=True, exist_ok=True)
                        _export.export_to_ltspice(transistor, out)
                        zip_file.write(out, arcname=f"{transistor_id}.asc")

                except Exception as e:
                    # Add error file to ZIP
                    error_content = f"Export failed for {transistor_id}: {e!s}"
                    zip_file.writestr(f"{transistor_id}_ERROR.txt", error_content)

        # Seek to beginning of buffer
        zip_buffer.seek(0)

        # Return ZIP file
        return StreamingResponse(
            io.BytesIO(zip_buffer.read()),
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename=transistor_exports_{request.format}.zip"
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch export failed: {e!s}")


# ---------------------------------------------------------------------------
# Upload / Import
# ---------------------------------------------------------------------------

@app.post("/api/transistors/upload")
async def upload_transistor(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Upload and import a transistor from a JSON file."""
    if not file.filename.endswith('.json'):
        raise HTTPException(status_code=400, detail="Only JSON files are supported")

    try:
        content = await file.read()
        # Write to temp file, load through the loader (which does numpy conversion)
        tmp = Path(tempfile.mktemp(suffix='.json'))
        tmp.write_bytes(content)
        transistor = _loader.load_from_json(tmp)
        tmp.unlink(missing_ok=True)

        _repo.save(transistor)
        return {
            "id": transistor.metadata.name,
            "message": "Transistor uploaded successfully",
            "name": transistor.metadata.name,
            "manufacturer": transistor.metadata.manufacturer,
            "type": transistor.metadata.type,
        }
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e!s}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Upload failed: {e!s}")


# ---------------------------------------------------------------------------
# Plot data endpoints
# ---------------------------------------------------------------------------

@app.get("/api/plots/channel/{transistor_id}")
async def plot_channel(
    transistor_id: str,
    component: str = Query("switch", pattern="^(switch|diode)$"),
) -> Dict[str, Any]:
    """Get channel characteristic plot data."""
    transistor = _get_transistor(transistor_id)
    return _plotting.plot_channel_characteristics(transistor, component=component)


@app.get("/api/plots/switching/{transistor_id}")
async def plot_switching(
    transistor_id: str,
    loss_type: str = Query("e_on", pattern="^(e_on|e_off|e_rr)$"),
    plot_type: str = Query("i_e", pattern="^(i_e|r_e|t_e)$"),
) -> Dict[str, Any]:
    """Get switching loss plot data."""
    transistor = _get_transistor(transistor_id)
    return _plotting.plot_switching_losses(transistor, loss_type=loss_type, plot_type=plot_type)


@app.get("/api/plots/soa/{transistor_id}")
async def plot_soa(transistor_id: str) -> Dict[str, Any]:
    """Get safe operating area plot data."""
    transistor = _get_transistor(transistor_id)
    return _plotting.plot_safe_operating_area(transistor)


@app.get("/api/plots/thermal/{transistor_id}")
async def plot_thermal(transistor_id: str) -> Dict[str, Any]:
    """Get thermal impedance plot data."""
    transistor = _get_transistor(transistor_id)
    return _plotting.plot_thermal_impedance(transistor)


@app.get("/api/plots/gate_charge/{transistor_id}")
async def plot_gate_charge(transistor_id: str) -> Dict[str, Any]:
    """Get gate charge plot data."""
    transistor = _get_transistor(transistor_id)
    return _plotting.plot_gate_charge(transistor)


# ---------------------------------------------------------------------------
# Curve management endpoints (Phase 1)
# ---------------------------------------------------------------------------

@app.get("/api/transistors/{transistor_id}/curves")
async def get_all_curves(transistor_id: str) -> Dict[str, Any]:
    """Get all curves for a transistor."""
    transistor = _get_transistor(transistor_id)

    def array_to_list(arr):
        """Convert numpy array to nested list for JSON."""
        if arr is None:
            return None
        return arr.tolist() if hasattr(arr, 'tolist') else arr

    return {
        "switch": {
            "channel_data": [
                {
                    "t_j": ch.t_j,
                    "v_g": ch.v_g,
                    "graph_v_i": array_to_list(ch.graph_v_i),
                }
                for ch in transistor.switch.channel_data
            ],
            "e_on_data": [
                {
                    "dataset_type": e.dataset_type,
                    "t_j": e.t_j,
                    "v_supply": e.v_supply,
                    "v_g": e.v_g,
                    "r_g": e.r_g,
                    "graph_i_e": array_to_list(e.graph_i_e),
                    "graph_r_e": array_to_list(e.graph_r_e),
                }
                for e in transistor.switch.e_on_data
            ],
            "e_off_data": [
                {
                    "dataset_type": e.dataset_type,
                    "t_j": e.t_j,
                    "v_supply": e.v_supply,
                    "v_g": e.v_g,
                    "r_g": e.r_g,
                    "graph_i_e": array_to_list(e.graph_i_e),
                    "graph_r_e": array_to_list(e.graph_r_e),
                }
                for e in transistor.switch.e_off_data
            ],
            "gate_charge_curves": [
                {
                    "v_supply": gc.v_supply,
                    "t_j": gc.t_j,
                    "i_channel": gc.i_channel,
                    "i_g": gc.i_g,
                    "graph_q_v": array_to_list(gc.graph_q_v),
                }
                for gc in transistor.switch.gate_charge_curves
            ],
            "soa": [
                {
                    "t_c": soa.t_c,
                    "time_pulse": soa.time_pulse,
                    "graph_i_v": array_to_list(soa.graph_i_v),
                }
                for soa in transistor.switch.soa
            ],
        },
        "diode": {
            "channel_data": [
                {
                    "t_j": ch.t_j,
                    "graph_v_i": array_to_list(ch.graph_v_i),
                }
                for ch in transistor.diode.channel_data
            ],
            "e_rr_data": [
                {
                    "dataset_type": e.dataset_type,
                    "t_j": e.t_j,
                    "v_supply": e.v_supply,
                    "v_g": e.v_g,
                    "r_g": e.r_g,
                    "graph_i_e": array_to_list(e.graph_i_e),
                }
                for e in transistor.diode.e_rr_data
            ],
        },
        "capacitances": {
            "c_oss": [
                {
                    "t_j": cap.t_j,
                    "graph_v_c": array_to_list(cap.graph_v_c),
                }
                for cap in transistor.c_oss
            ],
            "c_iss": [
                {
                    "t_j": cap.t_j,
                    "graph_v_c": array_to_list(cap.graph_v_c),
                }
                for cap in transistor.c_iss
            ],
            "c_rss": [
                {
                    "t_j": cap.t_j,
                    "graph_v_c": array_to_list(cap.graph_v_c),
                }
                for cap in transistor.c_rss
            ],
        },
    }


@app.post("/api/transistors/{transistor_id}/curves/channel")
async def add_channel_curve(
    transistor_id: str,
    curve_data: ChannelCurveCreate,
    component: str = Query("switch", pattern="^(switch|diode)$"),
) -> Dict[str, Any]:
    """Add a channel characteristic curve to switch or diode."""
    transistor = _get_transistor(transistor_id)

    # Convert lists to numpy 2D array
    graph_v_i = np.array([curve_data.v_data, curve_data.i_data], dtype=np.float64)

    # Create ChannelCharacteristics object
    new_curve = ChannelCharacteristics(
        t_j=curve_data.t_j,
        v_g=curve_data.v_g,
        graph_v_i=graph_v_i,
    )

    # Add to appropriate component
    if component == "switch":
        if new_curve.v_g is None:
            raise HTTPException(
                status_code=400,
                detail="v_g is mandatory for switch channel curves"
            )
        transistor.switch.channel_data.append(new_curve)
    else:  # diode
        transistor.diode.channel_data.append(new_curve)

    # Save transistor
    _repo.save(transistor)

    return {
        "message": f"Channel curve added to {component}",
        "curve_count": len(
            transistor.switch.channel_data if component == "switch"
            else transistor.diode.channel_data
        ),
    }


@app.post("/api/transistors/{transistor_id}/curves/switching/{loss_type}")
async def add_switching_loss_curve(
    transistor_id: str,
    loss_type: str,
    curve_data: SwitchingLossCreate,
) -> Dict[str, Any]:
    """Add switching loss data (e_on, e_off, e_rr)."""
    transistor = _get_transistor(transistor_id)

    if loss_type not in ("e_on", "e_off", "e_rr"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid loss_type '{loss_type}'. Must be 'e_on', 'e_off', or 'e_rr'"
        )

    # Convert graph data to numpy arrays
    graph_i_e = None
    graph_r_e = None
    graph_t_e = None

    if curve_data.graph_i_e:
        graph_i_e = np.array(curve_data.graph_i_e, dtype=np.float64)
    if curve_data.graph_r_e:
        graph_r_e = np.array(curve_data.graph_r_e, dtype=np.float64)
    if curve_data.graph_t_e:
        graph_t_e = np.array(curve_data.graph_t_e, dtype=np.float64)

    # Create SwitchingLossData object
    new_loss = SwitchingLossData(
        dataset_type=curve_data.dataset_type,
        t_j=curve_data.t_j,
        v_supply=curve_data.v_supply,
        v_g=curve_data.v_g,
        v_g_off=curve_data.v_g_off,
        r_g=curve_data.r_g,
        i_x=curve_data.i_x,
        e_x=curve_data.e_x,
        graph_i_e=graph_i_e,
        graph_r_e=graph_r_e,
        graph_t_e=graph_t_e,
        comment=curve_data.comment,
        measurement_date=curve_data.measurement_date,
        measurement_testbench=curve_data.measurement_testbench,
        commutation_device=curve_data.commutation_device,
        load_inductance=curve_data.load_inductance,
        commutation_inductance=curve_data.commutation_inductance,
    )

    # Add to appropriate list
    if loss_type == "e_on":
        transistor.switch.e_on_data.append(new_loss)
    elif loss_type == "e_off":
        transistor.switch.e_off_data.append(new_loss)
    else:  # e_rr
        transistor.diode.e_rr_data.append(new_loss)

    # Save transistor
    _repo.save(transistor)

    return {
        "message": f"Switching loss curve ({loss_type}) added",
        "curve_count": len(
            transistor.switch.e_on_data if loss_type == "e_on"
            else transistor.switch.e_off_data if loss_type == "e_off"
            else transistor.diode.e_rr_data
        ),
    }


@app.post("/api/transistors/{transistor_id}/curves/gate_charge")
async def add_gate_charge_curve(
    transistor_id: str,
    curve_data: GateChargeCurveCreate,
) -> Dict[str, Any]:
    """Add a gate charge curve."""
    transistor = _get_transistor(transistor_id)

    # Convert lists to numpy 2D array [charge, voltage]
    graph_q_v = np.array([curve_data.q_data, curve_data.v_data], dtype=np.float64)

    # Create GateChargeCurve object
    new_curve = GateChargeCurve(
        v_supply=curve_data.v_supply,
        t_j=curve_data.t_j,
        i_channel=curve_data.i_channel,
        i_g=curve_data.i_g,
        graph_q_v=graph_q_v,
    )

    transistor.switch.gate_charge_curves.append(new_curve)
    _repo.save(transistor)

    return {
        "message": "Gate charge curve added",
        "curve_count": len(transistor.switch.gate_charge_curves),
    }


@app.post("/api/transistors/{transistor_id}/curves/soa")
async def add_soa_curve(
    transistor_id: str,
    curve_data: SOACreate,
) -> Dict[str, Any]:
    """Add a Safe Operating Area curve."""
    transistor = _get_transistor(transistor_id)

    # Convert lists to numpy 2D array [voltage, current]
    graph_i_v = np.array([curve_data.v_data, curve_data.i_data], dtype=np.float64)

    # Create SOA object
    new_soa = SOA(
        t_c=curve_data.t_c,
        time_pulse=curve_data.time_pulse,
        graph_i_v=graph_i_v,
    )

    transistor.switch.soa.append(new_soa)
    _repo.save(transistor)

    return {
        "message": "SOA curve added",
        "curve_count": len(transistor.switch.soa),
    }


@app.post("/api/transistors/{transistor_id}/curves/capacitance/{cap_type}")
async def add_capacitance_curve(
    transistor_id: str,
    cap_type: str,
    curve_data: CapacitanceCurveCreate,
) -> Dict[str, Any]:
    """Add a voltage-dependent capacitance curve (C_oss, C_iss, C_rss)."""
    transistor = _get_transistor(transistor_id)

    if cap_type not in ("c_oss", "c_iss", "c_rss"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid cap_type '{cap_type}'. Must be 'c_oss', 'c_iss', or 'c_rss'"
        )

    # Convert lists to numpy 2D array [voltage, capacitance]
    graph_v_c = np.array([curve_data.v_data, curve_data.c_data], dtype=np.float64)

    # Create VoltageDependentCapacitance object
    new_cap = VoltageDependentCapacitance(
        t_j=curve_data.t_j,
        graph_v_c=graph_v_c,
    )

    # Add to appropriate list
    if cap_type == "c_oss":
        transistor.c_oss.append(new_cap)
    elif cap_type == "c_iss":
        transistor.c_iss.append(new_cap)
    else:  # c_rss
        transistor.c_rss.append(new_cap)

    _repo.save(transistor)

    return {
        "message": f"{cap_type.upper()} curve added",
        "curve_count": len(
            transistor.c_oss if cap_type == "c_oss"
            else transistor.c_iss if cap_type == "c_iss"
            else transistor.c_rss
        ),
    }


@app.delete("/api/transistors/{transistor_id}/curves/{component}/{curve_type}/{index}")
async def delete_curve(
    transistor_id: str,
    component: str,
    curve_type: str,
    index: int,
) -> Dict[str, Any]:
    """Delete a specific curve by component, type, and index."""
    transistor = _get_transistor(transistor_id)

    # Map component and curve_type to the appropriate list
    curve_map = {
        "switch": {
            "channel": transistor.switch.channel_data,
            "e_on": transistor.switch.e_on_data,
            "e_off": transistor.switch.e_off_data,
            "gate_charge": transistor.switch.gate_charge_curves,
            "soa": transistor.switch.soa,
        },
        "diode": {
            "channel": transistor.diode.channel_data,
            "e_rr": transistor.diode.e_rr_data,
        },
        "capacitance": {
            "c_oss": transistor.c_oss,
            "c_iss": transistor.c_iss,
            "c_rss": transistor.c_rss,
        },
    }

    if component not in curve_map:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid component '{component}'"
        )

    if curve_type not in curve_map[component]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid curve_type '{curve_type}' for component '{component}'"
        )

    curve_list = curve_map[component][curve_type]

    if not 0 <= index < len(curve_list):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid index {index}. Curve list has {len(curve_list)} items"
        )

    # Delete the curve
    del curve_list[index]
    _repo.save(transistor)

    return {
        "message": f"Curve deleted: {component}.{curve_type}[{index}]",
        "remaining_count": len(curve_list),
    }


@app.post("/api/transistors/{transistor_id}/curves/validate")
async def validate_curves(transistor_id: str) -> CurveValidationResult:
    """Validate all curves for a transistor."""
    transistor = _get_transistor(transistor_id)

    errors = []
    warnings = []

    # Validate switch channel curves
    for i, ch in enumerate(transistor.switch.channel_data):
        if ch.v_g is None:
            errors.append(f"Switch channel curve {i}: v_g is None")
        if ch.graph_v_i.shape[0] != 2:
            errors.append(f"Switch channel curve {i}: invalid graph shape")

    # Validate switching losses
    for i, loss in enumerate(transistor.switch.e_on_data):
        if loss.dataset_type == "single" and loss.e_x is None:
            errors.append(f"E_on curve {i}: single type but e_x is None")

    # Add more validation rules...

    statistics = {
        "switch_channel_curves": len(transistor.switch.channel_data),
        "diode_channel_curves": len(transistor.diode.channel_data),
        "e_on_curves": len(transistor.switch.e_on_data),
        "e_off_curves": len(transistor.switch.e_off_data),
        "e_rr_curves": len(transistor.diode.e_rr_data),
        "gate_charge_curves": len(transistor.switch.gate_charge_curves),
        "soa_curves": len(transistor.switch.soa),
    }

    return CurveValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        statistics=statistics,
    )


# ---------------------------------------------------------------------------
# Topology Calculator (Phase 4)
# ---------------------------------------------------------------------------

@app.post("/api/topology/calculate")
async def calculate_topology(request: TopologyCalculationRequest) -> Dict[str, Any]:
    """Calculate converter performance for a specific topology."""
    transistor = None
    if request.transistor_id:
        try:
            transistor = _get_transistor(request.transistor_id)
        except HTTPException:
            pass

    try:
        result = _topology_calc.calculate(
            request.topology,
            request.parameters.dict() if hasattr(request.parameters, 'dict') else request.parameters,
            transistor
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Calculation failed: {e!s}")


@app.post("/api/topology/compare")
async def compare_topologies(request: Dict[str, Any]) -> Dict[str, Any]:
    """Compare all three topologies for given parameters."""
    parameters = request.get('parameters', {})
    transistor_id = request.get('transistor_id')

    transistor = None
    if transistor_id:
        try:
            transistor = _get_transistor(transistor_id)
        except HTTPException:
            pass

    try:
        result = _topology_calc.compare_topologies(parameters, transistor)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {e!s}")


@app.get("/api/topology/recommendations")
async def topology_recommendations(
    topology: str = Query(..., description="Topology type"),
    v_in: float = Query(..., description="Input voltage (V)"),
    v_out: float = Query(..., description="Output voltage (V)"),
    p_out: float = Query(..., description="Output power (W)")
) -> Dict[str, Any]:
    """Get transistor recommendations for a topology."""
    if topology == "buck":
        v_required = v_in * 1.2
        i_required = (p_out / v_out) * 1.3
    elif topology == "boost":
        v_required = v_out * 1.2
        i_required = (p_out / v_in) * 1.3
    else:
        v_required = (v_in + v_out) * 1.2
        i_required = max(p_out / v_in, p_out / v_out) * 1.3

    all_names = _repo.list_all()
    suitable = []

    for name in all_names:
        t = _repo.get_by_name(name)
        if t is None:
            continue

        if (t.electrical_ratings.v_abs_max >= v_required and
            t.electrical_ratings.i_abs_max >= i_required):

            v_util = v_required / t.electrical_ratings.v_abs_max
            i_util = i_required / t.electrical_ratings.i_abs_max
            util_score = (v_util + i_util) / 2

            suitable.append({
                'name': t.metadata.name,
                'type': t.metadata.type,
                'manufacturer': t.metadata.manufacturer,
                'v_abs_max': t.electrical_ratings.v_abs_max,
                'i_abs_max': t.electrical_ratings.i_abs_max,
                'utilization': util_score * 100,
                'voltage_margin': (t.electrical_ratings.v_abs_max - v_required) / v_required * 100,
                'current_margin': (t.electrical_ratings.i_abs_max - i_required) / i_required * 100,
            })

    suitable.sort(key=lambda x: abs(x['utilization'] - 75))

    return {
        'topology': topology,
        'requirements': {
            'v_required': v_required,
            'i_required': i_required,
            'p_out': p_out,
        },
        'recommendations': suitable[:10],
        'total_suitable': len(suitable),
    }


# ---------------------------------------------------------------------------
# Settings & Advanced Features (Phase 5)
# ---------------------------------------------------------------------------

@app.get("/api/settings/user")
async def get_user_settings(user_id: str = "default") -> UserSettings:
    """Get user settings."""
    if user_id not in _user_settings:
        _user_settings[user_id] = UserSettings()
    return _user_settings[user_id]


@app.put("/api/settings/user")
async def update_user_settings(settings: UserSettings, user_id: str = "default") -> Dict[str, str]:
    """Update user settings."""
    _user_settings[user_id] = settings
    return {"message": "Settings updated successfully"}


@app.post("/api/settings/user/reset")
async def reset_user_settings(user_id: str = "default") -> Dict[str, str]:
    """Reset user settings to defaults."""
    _user_settings[user_id] = UserSettings()
    return {"message": "Settings reset to defaults"}


@app.post("/api/transistors/search/advanced")
async def advanced_search(request: AdvancedSearchRequest) -> Dict[str, Any]:
    """Advanced transistor search with filters, sorting, and pagination."""
    all_names = _repo.list_all()
    results = []

    for name in all_names:
        t = _repo.get_by_name(name)
        if t is None:
            continue

        # Apply filters
        filters = request.filters
        if filters.get('type') and t.metadata.type not in filters['type']:
            continue
        if 'v_abs_max' in filters:
            v_range = filters['v_abs_max']
            if 'min' in v_range and t.electrical_ratings.v_abs_max < v_range['min']:
                continue
            if 'max' in v_range and t.electrical_ratings.v_abs_max > v_range['max']:
                continue
        if 'i_abs_max' in filters:
            i_range = filters['i_abs_max']
            if 'min' in i_range and t.electrical_ratings.i_abs_max < i_range['min']:
                continue
            if 'max' in i_range and t.electrical_ratings.i_abs_max > i_range['max']:
                continue
        if filters.get('has_gate_charge') and not t.switch.gate_charge_curves:
            continue

        results.append({
            'name': t.metadata.name,
            'type': t.metadata.type,
            'manufacturer': t.metadata.manufacturer,
            'housing_type': t.metadata.housing_type,
            'v_abs_max': t.electrical_ratings.v_abs_max,
            'i_abs_max': t.electrical_ratings.i_abs_max,
            'i_cont': t.electrical_ratings.i_cont,
            'has_gate_charge': len(t.switch.gate_charge_curves) > 0,
            'has_switching_losses': len(t.switch.e_on_data) > 0 or len(t.switch.e_off_data) > 0,
        })

    # Sort results
    if request.sort_field:
        reverse = request.sort_order == 'desc'
        results.sort(key=lambda x: x.get(request.sort_field, 0), reverse=reverse)

    # Paginate
    total = len(results)
    start = (request.page - 1) * request.per_page
    end = start + request.per_page
    page_results = results[start:end]

    return {
        'results': page_results,
        'pagination': {
            'page': request.page,
            'per_page': request.per_page,
            'total': total,
            'pages': (total + request.per_page - 1) // request.per_page,
        }
    }


@app.post("/api/favorites/add")
async def add_favorite(transistor_id: str, user_id: str = "default") -> Dict[str, str]:
    """Add transistor to favorites."""
    settings = _user_settings.get(user_id, UserSettings())
    if transistor_id not in settings.favorite_transistors:
        settings.favorite_transistors.append(transistor_id)
        _user_settings[user_id] = settings
    return {"message": "Added to favorites"}


@app.post("/api/favorites/remove")
async def remove_favorite(transistor_id: str, user_id: str = "default") -> Dict[str, str]:
    """Remove transistor from favorites."""
    settings = _user_settings.get(user_id, UserSettings())
    if transistor_id in settings.favorite_transistors:
        settings.favorite_transistors.remove(transistor_id)
        _user_settings[user_id] = settings
    return {"message": "Removed from favorites"}


@app.get("/api/favorites/list")
async def list_favorites(user_id: str = "default") -> List[str]:
    """Get list of favorite transistors."""
    settings = _user_settings.get(user_id, UserSettings())
    return settings.favorite_transistors


# ==================== Phase 6: Archive Integration Endpoints ====================


@app.post("/api/import/plecs", response_model=PLECSImportResponse)
async def import_plecs(file: UploadFile) -> PLECSImportResponse:
    """Import transistors from PLECS XML semiconductor library file.

    :param file: Uploaded PLECS XML file
    :return: Import results with transistor IDs
    """
    from transistordatabase.plecs_importer import import_plecs_xml
    import tempfile

    errors = []
    warnings = []
    transistor_ids = []

    try:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xml') as tmp:
            content = await file.read()
            tmp.write(content)
            tmp.flush()

            # Import transistors
            transistors = import_plecs_xml(tmp.name)

            # Save to repository
            for transistor in transistors:
                try:
                    _repo.save(transistor)
                    transistor_ids.append(transistor.metadata.name)
                except Exception as e:
                    errors.append(f"Failed to save {transistor.metadata.name}: {str(e)}")

            return PLECSImportResponse(
                success=len(transistor_ids) > 0,
                transistors_imported=len(transistor_ids),
                transistor_ids=transistor_ids,
                errors=errors,
                warnings=warnings
            )

    except Exception as e:
        return PLECSImportResponse(
            success=False,
            transistors_imported=0,
            transistor_ids=[],
            errors=[str(e)]
        )


@app.post("/api/analytical/biela")
async def calculate_biela(request: BielaModelRequest) -> Dict[str, Any]:
    """Calculate switching losses using Biela analytical model.

    :param request: Biela model parameters and operating conditions
    :return: Calculated energies and optional curve data
    """
    from transistordatabase.analytical_models import BielaModel, BielaModelParams

    # Create model
    params = BielaModelParams(
        c_oss=request.c_oss,
        q_g=request.q_g,
        v_plateau=request.v_plateau,
        r_g_on=request.r_g_on,
        r_g_off=request.r_g_off,
        v_driver_on=request.v_driver_on,
        v_driver_off=request.v_driver_off
    )
    model = BielaModel(params)

    # Calculate at single operating point
    e_on = model.calc_e_on(request.v_dc, request.i_load, request.t_j)
    e_off = model.calc_e_off(request.v_dc, request.i_load, request.t_j)

    result = {
        'operating_point': {
            'v_dc': request.v_dc,
            'i_load': request.i_load,
            't_j': request.t_j,
        },
        'results': {
            'e_on': e_on,
            'e_off': e_off,
            'e_total': e_on + e_off,
        }
    }

    # If curve requested, calculate over current range
    if request.i_min is not None and request.i_max is not None:
        currents = np.linspace(request.i_min, request.i_max, request.i_points)
        e_on_curve, e_off_curve, e_total_curve = model.calc_switching_loss_curve(
            request.v_dc, currents, request.t_j
        )

        result['curve'] = {
            'currents': currents.tolist(),
            'e_on': e_on_curve.tolist(),
            'e_off': e_off_curve.tolist(),
            'e_total': e_total_curve.tolist(),
        }

    return result


@app.post("/api/analytical/gate_charge")
async def calculate_gate_charge(request: GateChargeModelRequest) -> Dict[str, Any]:
    """Calculate switching times using gate charge model.

    :param request: Gate charge parameters
    :return: Turn-on and turn-off timing results
    """
    from transistordatabase.analytical_models import GateChargeModel, GateChargeModelParams

    # Create model
    params = GateChargeModelParams(
        q_gs=request.q_gs,
        q_gd=request.q_gd,
        q_g=request.q_g,
        v_th=request.v_th,
        v_plateau=request.v_plateau,
        r_g=request.r_g,
        r_g_int=request.r_g_int
    )
    model = GateChargeModel(params)

    # Calculate turn-on times
    turn_on = model.calc_turn_on_times(request.v_driver)

    # Calculate turn-off times
    turn_off = model.calc_turn_off_times(request.v_driver, request.v_off)

    return {
        'parameters': {
            'q_gs': request.q_gs,
            'q_gd': request.q_gd,
            'q_g': request.q_g,
            'v_th': request.v_th,
            'v_plateau': request.v_plateau,
            'r_g_total': request.r_g + request.r_g_int,
        },
        'turn_on_times': {
            't_delay': turn_on['t_delay'] * 1e9,  # Convert to ns
            't_rise': turn_on['t_rise'] * 1e9,
            't_fall_v': turn_on['t_fall_v'] * 1e9,
            't_total': turn_on['t_total'] * 1e9,
        },
        'turn_off_times': {
            't_delay': turn_off['t_delay'] * 1e9,
            't_rise_v': turn_off['t_rise_v'] * 1e9,
            't_fall_i': turn_off['t_fall_i'] * 1e9,
            't_total': turn_off['t_total'] * 1e9,
        }
    }


@app.post("/api/dpt/generate_netlist")
async def generate_dpt_netlist(
    transistor_id: str,
    config: DPTConfigRequest
) -> Dict[str, str]:
    """Generate LTspice Double Pulse Test netlist for a transistor.

    :param transistor_id: ID of transistor to test
    :param config: DPT configuration parameters
    :return: Generated netlist as text
    """
    from transistordatabase.utils.ltspice_dpt import LTspiceDPT, DPTConfig

    transistor = _get_transistor(transistor_id)

    # Create DPT configuration
    dpt_config = DPTConfig(
        v_dc=config.v_dc,
        i_target=config.i_target,
        l_load=config.l_load,
        r_g_on=config.r_g_on,
        r_g_off=config.r_g_off,
        v_g_on=config.v_g_on,
        v_g_off=config.v_g_off,
        t_dead=config.t_dead,
        t_pulse2=config.t_pulse2
    )

    # Generate netlist
    dpt = LTspiceDPT()
    netlist = dpt.generate_netlist(transistor, dpt_config)

    return {
        'netlist': netlist,
        'transistor_name': transistor.metadata.name,
        'config': {
            'v_dc': config.v_dc,
            'i_target': config.i_target,
            'r_g_on': config.r_g_on,
            'r_g_off': config.r_g_off,
        }
    }


@app.get("/api/transistors/{transistor_id}/dpt_data")
async def get_dpt_data(transistor_id: str) -> Dict[str, Any]:
    """Get DPT measurement data for a transistor.

    :param transistor_id: Transistor ID
    :return: DPT data if available
    """
    transistor = _get_transistor(transistor_id)

    # Extract DPT data from switching loss data
    dpt_data = {
        'has_e_on': len(transistor.switch.e_on_data) > 0,
        'has_e_off': len(transistor.switch.e_off_data) > 0,
        'has_e_rr': len(transistor.diode.e_rr_data) > 0 if transistor.diode else False,
    }

    # Get measurement metadata from first dataset if available
    if transistor.switch.e_on_data:
        first = transistor.switch.e_on_data[0]
        dpt_data['metadata'] = {
            't_j': first.t_j,
            'v_supply': first.v_supply,
            'v_g': first.v_g,
            'r_g': first.r_g,
            'comment': first.comment or '',
        }

    return dpt_data


@app.post("/api/transistors/{transistor_id}/dpt_validate")
async def validate_dpt_data(transistor_id: str) -> Dict[str, Any]:
    """Validate DPT measurement data against datasheet specifications.

    :param transistor_id: Transistor ID
    :return: Validation results
    """
    transistor = _get_transistor(transistor_id)

    errors = []
    warnings = []

    # Check if transistor has switching loss data
    if not transistor.switch.e_on_data and not transistor.switch.e_off_data:
        errors.append("No switching loss data available for validation")
        return {'valid': False, 'errors': errors, 'warnings': warnings}

    # Check voltage consistency
    v_supplies = set()
    for loss_data in transistor.switch.e_on_data + transistor.switch.e_off_data:
        v_supplies.add(loss_data.v_supply)

    if transistor.electrical_ratings.v_abs_max > 0:
        for v_supply in v_supplies:
            if v_supply > transistor.electrical_ratings.v_abs_max:
                errors.append(
                    f"Supply voltage {v_supply}V exceeds rated maximum "
                    f"{transistor.electrical_ratings.v_abs_max}V"
                )

    # Check current levels
    max_currents = []
    for loss_data in transistor.switch.e_on_data:
        if loss_data.graph_i_e and len(loss_data.graph_i_e[0]) > 0:
            max_currents.append(max(loss_data.graph_i_e[0]))

    if max_currents and transistor.electrical_ratings.i_abs_max > 0:
        max_test_current = max(max_currents)
        if max_test_current > transistor.electrical_ratings.i_abs_max:
            warnings.append(
                f"Test current {max_test_current}A exceeds rated maximum "
                f"{transistor.electrical_ratings.i_abs_max}A"
            )

    # Check temperature range
    temperatures = set()
    for loss_data in transistor.switch.e_on_data + transistor.switch.e_off_data:
        temperatures.add(loss_data.t_j)

    if temperatures:
        if min(temperatures) < -55:
            warnings.append(f"Test temperature {min(temperatures)}°C is very low")
        if max(temperatures) > transistor.electrical_ratings.t_j_max:
            errors.append(
                f"Test temperature {max(temperatures)}°C exceeds rated maximum "
                f"{transistor.electrical_ratings.t_j_max}°C"
            )

    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'statistics': {
            'v_supply_range': [min(v_supplies), max(v_supplies)] if v_supplies else [0, 0],
            'i_max_tested': max(max_currents) if max_currents else 0,
            't_j_range': [min(temperatures), max(temperatures)] if temperatures else [0, 0],
        }
    }


# ==================== Switching-Pair Endpoints ====================


@app.get("/api/pairs/")
async def list_pairs(
    pair_type: str | None = None,
    device: str | None = None,
    manufacturer: str | None = None,
    v_min: float | None = None,
    v_max: float | None = None,
    source: str | None = None,
    page: int = 1,
    per_page: int = 20,
) -> Dict[str, Any]:
    """List switching pairs with filtering and pagination.

    Query parameters:
    - pair_type: Filter by pair type (mosfet_self, mosfet_plus_diode, etc.)
    - device: Filter by device ID
    - manufacturer: Filter by manufacturer
    - v_min: Minimum blocking voltage (V)
    - v_max: Maximum blocking voltage (V)
    - source: Filter by source (datasheet, dpt_measurement, etc.)
    - page: Page number (default: 1)
    - per_page: Results per page (default: 20)

    Returns: Dict with total count, pagination info, and list of pair summaries.
    """
    filters = {}
    if pair_type:
        filters['pair_type'] = pair_type
    if device:
        filters['device'] = device
    if manufacturer:
        filters['manufacturer'] = manufacturer
    if v_min is not None:
        filters['v_min'] = v_min
    if v_max is not None:
        filters['v_max'] = v_max
    if source:
        filters['source'] = source

    all_pairs = _pair_repo.list_pairs(filters=filters if filters else None)

    # Pagination
    total = len(all_pairs)
    start = (page - 1) * per_page
    end = start + per_page
    paginated = all_pairs[start:end]

    return {
        "total": total,
        "page": page,
        "per_page": per_page,
        "pairs": paginated,
    }


@app.get("/api/pairs/{pair_id}")
async def get_pair(pair_id: str) -> Dict[str, Any]:
    """Get full switching pair data by ID.

    Returns: Complete SwitchingPair object with all metadata, electrical ratings,
    switching data, conduction curves, and thermal properties.
    """
    try:
        pair = _pair_repo.load_pair(pair_id)
        return pair.to_dict()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Switching pair not found: {pair_id}")


@app.get("/api/pairs/{pair_id}/validate")
async def validate_pair(pair_id: str) -> Dict[str, Any]:
    """Validate a switching pair and return quality metrics.

    Returns: Validation result with errors, warnings, and quality_score (0-100).
    """
    try:
        pair = _pair_repo.load_pair(pair_id)
        return pair.validate()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Switching pair not found: {pair_id}")


@app.get("/api/devices/")
async def list_unique_devices(
    type: str | None = None,
    manufacturer: str | None = None,
) -> Dict[str, Any]:
    """List unique devices derived from all switching pairs.

    Query parameters:
    - type: Filter by device type (MOSFET, IGBT, Diode, etc.)
    - manufacturer: Filter by manufacturer

    Returns: Dict with total count and list of unique devices.
    """
    devices = _pair_repo.get_unique_devices()

    if type:
        devices = [d for d in devices if d.get('type', '').lower() == type.lower()]
    if manufacturer:
        devices = [d for d in devices if d.get('manufacturer', '').lower() == manufacturer.lower()]

    return {"total": len(devices), "devices": devices}


@app.get("/api/devices/{device_id}/pairs")
async def get_device_pairs(device_id: str) -> Dict[str, Any]:
    """Get all switching pairs containing a specific device.

    Returns: Dict with device_id, total count, and list of pair summaries.
    """
    results = _pair_repo.search_by_device(device_id)
    return {"device_id": device_id, "total": len(results), "pairs": results}


@app.get("/api/pairs/export/csv")
async def export_pairs_csv(
    pair_type: str | None = None,
    v_min: float | None = None,
    v_max: float | None = None,
) -> StreamingResponse:
    """Export switching pairs as ScDataGeneric CSV for ntbee2.

    Query parameters:
    - pair_type: Filter by pair type
    - v_min: Minimum voltage (V)
    - v_max: Maximum voltage (V)

    Returns: CSV file with columns: name, blockingVoltage, pair_type, devices,
    thermal resistance, cost, weight, source, quality_score.
    """
    filters = {}
    if pair_type:
        filters['pair_type'] = pair_type
    if v_min is not None:
        filters['v_min'] = v_min
    if v_max is not None:
        filters['v_max'] = v_max

    pairs = _pair_repo.list_pairs(filters=filters if filters else None)

    # CSV columns matching ScDataGeneric format
    columns = [
        'name', 'blockingVoltage', 'pair_type', 'high_side_device', 'low_side_device',
        'high_side_count', 'low_side_count',
        'forwardThermalResistance', 'reverseThermalResistance',
        'cost', 'weight', 'source', 'quality_score',
    ]

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()

    for pair_summary in pairs:
        try:
            pair = _pair_repo.load_pair(pair_summary['pair_id'])
            pair_dict = pair.to_dict()

            hs = pair_dict.get('high_side', {})
            ls = pair_dict.get('low_side', {})

            row = {
                'name': pair_dict.get('pair_id', ''),
                'blockingVoltage': hs.get('electrical_ratings', {}).get('v_abs_max', 0),
                'pair_type': pair_dict.get('pair_type', ''),
                'high_side_device': hs.get('device_id', ''),
                'low_side_device': ls.get('device_id', ''),
                'high_side_count': hs.get('count', 1),
                'low_side_count': ls.get('count', 1),
                'forwardThermalResistance': hs.get('thermal_properties', {}).get('r_th_jc', 0),
                'reverseThermalResistance': hs.get('thermal_properties', {}).get('r_th_cs', 0),
                'cost': hs.get('metadata', {}).get('cost', 0) or 0,
                'weight': hs.get('metadata', {}).get('weight', 0) or 0,
                'source': pair_dict.get('switching_data', {}).get('source', ''),
                'quality_score': pair_dict.get('migration', {}).get('quality_score', 0),
            }
            writer.writerow(row)
        except Exception as e:
            logger.warning(f"Error exporting pair {pair_summary.get('pair_id')}: {e}")

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=switching_pairs.csv"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)


# ---------------------------------------------------------------------------
# Static file serving for the Vue 3 frontend (webgui flavour)
# ---------------------------------------------------------------------------

def _frontend_dist() -> Path | None:
    """Locate the Vue 3 dist/ directory (works in-source and PyInstaller).

    :return: Path to the dist/ directory, or None if not found.
    """
    import sys
    if getattr(sys, "frozen", False):
        # PyInstaller onefile: files are extracted to sys._MEIPASS
        candidate = Path(sys._MEIPASS) / "transistordatabase" / "gui_web" / "dist"
    else:
        candidate = Path(__file__).parent.parent / "dist"
    return candidate if candidate.exists() else None


_dist = _frontend_dist()
if _dist is not None:
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=_dist, html=True), name="frontend")
