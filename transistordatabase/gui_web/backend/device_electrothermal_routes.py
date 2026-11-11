"""Read-only native inventory and fixed-sink device feedback through repository injection."""
from __future__ import annotations

from hashlib import sha256
from decimal import Decimal
import json
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import Field, field_validator

from transistordatabase.core.repository import JsonTransistorLoader, RepositoryPathError
from transistordatabase.core.services import TransistorRepository
from transistordatabase.core.serialization import _decode, SCHEMA
from transistordatabase.condition_tables import MeasuredTable
from transistordatabase.device_electrothermal import (
    DeviceThermalResourceError, analyze_device_electrothermal,
    _number, _graph, _project, _energy_context, _conditions, _FAMILIES,
    _ENERGY_NUMERIC, _ENERGY_TEXT, _bounded_output,
)
from transistordatabase.gui_web.backend.finite_json_route import FiniteJsonRoute, binary64_number
from transistordatabase.gui_web.backend.parameter_routes import Strict

MAX_SOURCE_BYTES = 16 * 1024 * 1024
MAX_INVENTORY_RECORDS = 512


def device_repository() -> TransistorRepository:
    """Resolve the existing application repository or a test-only injected provider."""
    raise RuntimeError('Configure the native-device repository dependency')


def _stored_float(literal):
    return binary64_number(Decimal(literal))


def _leaves(values):
    pending = [values]
    while pending:
        value = pending.pop()
        if isinstance(value, (list, tuple)):
            pending.extend(reversed(value))
        else:
            yield value


def _validate_stored_casts(document):
    """Check native numeric casts through existing codec nodes, not opaque metadata."""
    if not isinstance(document, dict) or document.get('schema') != SCHEMA:
        return
    pending = [document.get('transistor')]
    while pending:
        value = pending.pop()
        if not isinstance(value, dict):
            continue
        kind = value.get('kind')
        if kind == 'numpy-scalar':
            decoded = _decode(value)
            if decoded.dtype.kind in 'iuf':
                original = Decimal(value['value'])
                if ((original == 0) != (decoded == 0)
                        or (original < 0) != (decoded < 0)
                        or (decoded.dtype.kind in 'iu' and original != int(decoded))):
                    raise ValueError('Stored numeric dtype changes an original numeric identity')
        elif kind == 'ndarray':
            decoded = _decode(value)
            original_values = _decode(value['values'])
            for original, converted in zip(_leaves(original_values), decoded.flat, strict=True):
                if ((original == 0) != (converted == 0)
                        or (original < 0) != (converted < 0)
                        or (decoded.dtype.kind in 'iu' and original != int(converted))):
                    raise ValueError('Stored array dtype changes an original sign or nonzero sample')
            pending.append(value['values'])
        elif kind in ('aggregate', 'dataclass'):
            pending.extend(value.get('fields', {}).values())
        elif kind == 'mapping':
            pending.extend(value.get('values', {}).values())
        elif kind in ('list', 'tuple'):
            pending.extend(value.get('values', []))


def _snapshot(repository, name):
    storage = getattr(repository, '_storage_path', None)
    source = None
    if callable(storage):
        path = storage(name)
        try:
            with path.open('rb') as stream:
                raw = stream.read(MAX_SOURCE_BYTES + 1)
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail='The selected device does not exist') from error
        if len(raw) > MAX_SOURCE_BYTES:
            raise DeviceThermalResourceError('Native device source exceeds16MiB; no file was changed')
        document = json.loads(raw, parse_float=_stored_float, parse_constant=_stored_float)
        _validate_stored_casts(document)
        device = JsonTransistorLoader().load_from_dict(document)
        source = {'kind': 'repository-file', 'sha256': sha256(raw).hexdigest(),
                  'byte_length': len(raw), 'name': name,
                  'format': document.get('schema', 'legacy-flat'),
                  'scope': 'Original persisted bytes parsed once for this source snapshot'}
    else:
        device = repository.get_by_name(name)
    if device is None:
        raise HTTPException(status_code=404, detail='The selected device does not exist')
    if device.metadata.name != name:
        raise HTTPException(status_code=409, detail='Stored device identity differs from the selected repository name')
    return device, source


class CurveFamily(Strict):
    """Explicit selected source rows with named known and unknown conditions."""

    indices: Annotated[list[int], Field(max_length=32)]
    conditions: dict[str, float | str | None]

    @field_validator('conditions', mode='before')
    @classmethod
    def preserve_condition_values(cls, values):
        """Preserve exact declared numeric context before typed conversion."""
        if not isinstance(values, dict) or len(values) > 11:
            raise ValueError('Supply bounded named curve conditions')
        result = {}
        for name, value in values.items():
            if name not in _ENERGY_NUMERIC + _ENERGY_TEXT:
                raise ValueError('Unknown curve condition name')
            if value is None or name in _ENERGY_TEXT:
                result[name] = value
            else:
                result[name] = binary64_number(value)
        return result


class DeviceThermalRequest(Strict):
    """One declared switch-junction operating point and bounded solver controls."""

    transistor_id: str = Field(min_length=1, max_length=200)
    expected_file_sha256: str | None = Field(None, pattern=r'^[0-9a-fA-F]{64}$')
    selected: bool = False
    switching_policy: Literal['held-current-pwm']
    source: str = Field(min_length=1, max_length=2000)
    current_a: float = Field(ge=0)
    duty: float = Field(ge=0, le=1)
    frequency_hz: float = Field(ge=0)
    sink_temperature_c: float
    resistance_k_w: float = Field(ge=0)
    initial_temperature_c: float
    tolerance_k: float = Field(1e-8, gt=0)
    max_iterations: int = Field(200, ge=1, le=10000)
    relaxation: float = Field(1, gt=0, le=1)
    channel: CurveFamily | None = None
    turn_on: CurveFamily | None = None
    turn_off: CurveFamily | None = None

    @field_validator('current_a', 'duty', 'frequency_hz', 'sink_temperature_c',
                     'resistance_k_w', 'initial_temperature_c', 'tolerance_k',
                     'relaxation', mode='before')
    @classmethod
    def preserve_scalar_identity(cls, value):
        """Reject original-token precision loss before active/inactive classification."""
        return binary64_number(value)


router = APIRouter(prefix='/api/thermal/device', tags=['Native device electrothermal'], route_class=FiniteJsonRoute)


def _inventory(device, source):
    count = sum(len(getattr(device.switch, attribute)) for attribute in _FAMILIES.values())
    if count > MAX_INVENTORY_RECORDS:
        raise DeviceThermalResourceError('Device inventory exceeds512 source records')
    families = {}
    for family, attribute in _FAMILIES.items():
        rows = []
        for index, record in enumerate(getattr(device.switch, attribute)):
            row = {'index': index, 'temperature_c': None, 'recorded_conditions': None,
                   'representation': 'graph_v_i' if family == 'channel' else record.dataset_type,
                   'current_domain_a': None, 'sample_count': None, 'status': 'unavailable', 'diagnostic': None}
            try:
                row['temperature_c'] = _number(record.t_j, 'Recorded temperature')
                row['recorded_conditions'] = (_conditions({'v_g': record.v_g}, ('v_g',))
                                              if family == 'channel' else _energy_context(record))
                if family != 'channel' and record.dataset_type == 'single':
                    current = _number(record.i_x, 'Single measurement current', minimum=0)
                    _number(record.e_x, 'Single measurement energy', minimum=0)
                    row.update(current_domain_a=[current, current], sample_count=1)
                else:
                    if family != 'channel' and record.dataset_type != 'graph_i_e':
                        raise ValueError('This workflow supports current-energy curves or exact-current single measurements')
                    graph = _project(_graph(record, family))
                    x, y = (graph[1], graph[0]) if family == 'channel' else graph
                    table = MeasuredTable('channel' if family == 'channel' else 'energy', x, y, {}, 'Native inventory')
                    row.update(current_domain_a=[float(table.x[0]), float(table.x[-1])], sample_count=len(table.x))
                row['status'] = 'available'
            except (ValueError, TypeError, OverflowError) as error:
                row['diagnostic'] = str(error)
            rows.append(row)
        families[family] = rows
    return _bounded_output({'transistor_id': device.metadata.name, 'repository_source': source, 'families': families,
        'scope': 'Source summaries only; select explicit temperature families with fixed conditions. No graphs are repeated here.'})


@router.get('/inventory/{transistor_id}')
def native_inventory(transistor_id: str, repository: TransistorRepository = Depends(device_repository)):
    """List bounded source metadata without choosing device conditions for the user."""
    try:
        device, source = _snapshot(repository, transistor_id)
        return _inventory(device, source)
    except DeviceThermalResourceError as error:
        raise HTTPException(status_code=413, detail=str(error)) from error
    except (RepositoryPathError, ValueError, TypeError, OverflowError, RecursionError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.post('/analyze')
def native_analyze(request: DeviceThermalRequest, repository: TransistorRepository = Depends(device_repository)):
    """Resolve one immutable source snapshot and call the unchanged thermal solver."""
    try:
        device, source = _snapshot(repository, request.transistor_id)
        if request.expected_file_sha256 is not None and (source is None or source['sha256'].lower() != request.expected_file_sha256.lower()):
            raise HTTPException(status_code=409, detail='The device source changed or has no raw-file identity; refresh its inventory before calculation')
        values = request.model_dump(exclude={'transistor_id', 'expected_file_sha256'})
        return analyze_device_electrothermal(device, values, repository_source=source)
    except DeviceThermalResourceError as error:
        raise HTTPException(status_code=413, detail=str(error)) from error
    except (RepositoryPathError, ValueError, TypeError, OverflowError, RecursionError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
