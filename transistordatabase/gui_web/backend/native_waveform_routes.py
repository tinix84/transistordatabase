"""Bounded HTTP transport for supplied native-device waveform evaluation.

Schema and JSON-parser errors retain standard HTTP 422 validation details.
Scientific failures and all resource limits expose a stable error code.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import Field, field_validator

from transistordatabase.core.repository import RepositoryPathError
from transistordatabase.core.services import TransistorRepository
from transistordatabase.device_electrothermal import DeviceThermalResourceError
from transistordatabase.gui_web.backend.device_electrothermal_routes import (
    _snapshot, device_repository,
)
from transistordatabase.gui_web.backend.finite_json_route import (
    FiniteJsonRoute, binary64_number,
)
from transistordatabase.gui_web.backend.parameter_routes import Strict
from transistordatabase.native_waveform import (
    MAX_REQUEST_BYTES, MAX_SAMPLES, NativeWaveformError, evaluate_native_waveform,
)


class NativeWaveformRequest(Strict):
    """Resolve a database identity and preserve the pure operation's two inputs."""

    device_id: str = Field(min_length=1, max_length=200)
    selection: dict[str, Any]
    case: dict[str, Any]

    @field_validator("device_id")
    @classmethod
    def single_storage_identity(cls, value):
        """Preserve a literal identity while rejecting paths and control characters."""
        if (not value.strip() or value in {".", ".."}
                or any(character in value for character in "/\\:")
                or any(ord(character) < 32 for character in value)):
            raise ValueError("device_id must be one database name without path syntax")
        return value


class NativeWaveformRoute(FiniteJsonRoute):
    """Enforce the new operation's raw-byte budget before JSON parsing."""

    max_body_bytes = MAX_REQUEST_BYTES

    def get_route_handler(self):
        """Give raw-body limits the same code as scientific resource errors."""
        original = super().get_route_handler()

        async def handler(request):
            response = await original(request)
            if response.status_code == 413:
                return JSONResponse(status_code=413, content={"detail": {
                    "code": "resource-limit",
                    "message": f"The request exceeds the {self.max_body_bytes} byte limit",
                }})
            return response

        return handler


def _http_numbers(value, *, depth=0, counter=None):
    """Normalize preserved JSON numbers within bounded containers.

    Integral tokens retain their type for schema versions and record indices.
    Decimal tokens are checked before conversion to the pure binary64 contract.
    The scientific module owns every field and condition validation.
    """
    if counter is None:
        counter = [0]
    counter[0] += 1
    if depth > 12 or counter[0] > 100000:
        raise NativeWaveformError("resource-limit", "The JSON structure exceeds the native waveform budget")
    if isinstance(value, dict):
        if len(value) > 32:
            raise NativeWaveformError("resource-limit", "A JSON object exceeds the native waveform field budget")
        return {key: _http_numbers(item, depth=depth + 1, counter=counter)
                for key, item in value.items()}
    if isinstance(value, list):
        if len(value) > MAX_SAMPLES:
            raise NativeWaveformError("resource-limit", "A JSON vector exceeds the native waveform sample budget")
        return [_http_numbers(item, depth=depth + 1, counter=counter) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (Decimal, float, int)):
        checked = binary64_number(value)
        return value if type(value) is int else checked
    raise NativeWaveformError("invalid-input", "Supply JSON numbers, booleans, text, arrays, objects, or null")


router = APIRouter(
    prefix="/api/waveform/native", tags=["Native device waveforms"],
    route_class=NativeWaveformRoute,
)


@router.post("/analyze")
def analyze_native_waveform(
    request: NativeWaveformRequest,
    repository: TransistorRepository = Depends(device_repository),
):
    """Return the pure evaluation record without an additional calculation."""
    try:
        counter = [0]
        selection = _http_numbers(request.selection, counter=counter)
        case = _http_numbers(request.case, counter=counter)
        device, _source = _snapshot(repository, request.device_id)
        return evaluate_native_waveform(device, selection, case)
    except NativeWaveformError as error:
        status = 413 if error.code == "resource-limit" else 422
        raise HTTPException(status_code=status, detail={
            "code": error.code, "message": str(error),
        }) from error
    except DeviceThermalResourceError as error:
        raise HTTPException(status_code=413, detail={
            "code": "resource-limit", "message": str(error),
        }) from error
    except (RepositoryPathError, ValueError, TypeError, OverflowError, RecursionError) as error:
        raise HTTPException(status_code=422, detail={
            "code": "invalid-input", "message": str(error),
        }) from error

