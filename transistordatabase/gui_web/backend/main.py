"""
FastAPI backend for Transistor Database Web Application.

Uses the real core services and JsonTransistorRepository for persistent
storage instead of in-memory placeholder dicts.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

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
)
from transistordatabase.core.repository import (
    JsonTransistorRepository,
    JsonTransistorLoader,
)

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

_services = ConcreteServiceFactory.create_all_services()
_validation: ValidationService = _services['validation']
_comparison: ComparisonService = _services['comparison']
_export: ExportService = _services['export']
_plotting: PlottingService = _services['plotting']
_loader = JsonTransistorLoader()


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
    """Compare multiple transistors."""
    if len(transistor_ids) < 2:
        raise HTTPException(status_code=400, detail="At least 2 transistors required")
    transistors = [_get_transistor(tid) for tid in transistor_ids]
    return _comparison.compare_transistors(transistors)


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
