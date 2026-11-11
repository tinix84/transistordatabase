"""One switch junction from explicit native curve families and a fixed sink."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import math
from numbers import Real

import numpy as np

from transistordatabase.condition_tables import MeasuredTable
from transistordatabase.core.serialization import _encode
from transistordatabase.thermal import electrothermal
from transistordatabase.rg_energy_access import NUMERIC_CONDITIONS, TEXT_CONDITIONS, _conditions as switching_conditions

MAX_RECORDS = 32
MAX_GRAPH_POINTS = 10000
MAX_SELECTED_POINTS = 1000000
MAX_OUTPUT_BYTES = 4 * 1024 * 1024
MAX_COMMENT_CHARS = 65536
_ENERGY_NUMERIC = (*NUMERIC_CONDITIONS, 'r_g')
_ENERGY_TEXT = (*TEXT_CONDITIONS, 'measurement_testbench')
_FAMILIES = {'channel': 'channel_data', 'turn_on': 'e_on_data', 'turn_off': 'e_off_data'}
_FIELDS = {'selected', 'switching_policy', 'source', 'current_a', 'duty', 'frequency_hz',
           'sink_temperature_c', 'resistance_k_w', 'initial_temperature_c',
           'tolerance_k', 'max_iterations', 'relaxation', *_FAMILIES}


class DeviceThermalResourceError(ValueError):
    """A requested source snapshot exceeds the bounded native-curve work budget."""


def _number(value, label, *, minimum=None, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f'{label} requires an actual finite number')
    try:
        result = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError(f'{label} exceeds the finite SI range') from error
    if not math.isfinite(result) or result != value:
        raise ValueError(f'{label} exceeds the supported nonzero finite SI range')
    if (minimum is not None and result < minimum) or (positive and result <= 0):
        raise ValueError(f'{label} is outside its physical domain')
    return result


def _text(value, label):
    if not isinstance(value, str) or not value.strip() or len(value) > 2000:
        raise ValueError(f'{label} requires nonblank source text of at most2000 characters')
    value.encode('utf-8')
    return value


def _conditions(values, names):
    if not isinstance(values, dict) or set(values) != set(names):
        raise ValueError('Supply every named family condition, including explicit null for unknowns: ' + ', '.join(names))
    return {key: None if values[key] is None else _number(values[key], key) for key in names}


def _energy_context(record):
    values = {name: getattr(record, name) for name in NUMERIC_CONDITIONS}
    for name, value in values.items():
        if value is not None:
            _number(value, name)
    values['commutation_device'] = record.commutation_device
    values.update({name: getattr(record.evidence, name, None) for name in ('convention', 'event', 'method')})
    values = switching_conditions(values)
    values['r_g'] = None if record.r_g is None else _number(record.r_g, 'Recorded gate resistance', minimum=0)
    bench = record.measurement_testbench
    values['measurement_testbench'] = None if bench is None else _text(bench, 'Measurement testbench')
    return values


def _energy_selection(values):
    required = {'v_g', 'v_g_off', 'r_g', 'v_supply'}
    allowed = set(_ENERGY_NUMERIC + _ENERGY_TEXT)
    if not isinstance(values, dict) or len(values) > len(allowed) or not required <= set(values) <= allowed:
        raise ValueError('Declare the four gate/supply conditions; optional fixed hardware and evidence selectors must use native names')
    return {key: None if value is None else _text(value, key) if key in _ENERGY_TEXT else _number(value, key)
            for key, value in values.items()}


def _match(recorded, requested):
    if any(key not in recorded or recorded[key] != value for key, value in requested.items()):
        raise ValueError('Selected fixed numeric, hardware, event and method conditions must match exactly')


def _comment(record):
    value = getattr(record, 'comment', None)
    if value is None or value == '':
        return {'source_comment': None, 'source_comment_preview': None, 'source_comment_sha256': None, 'source_comment_truncated': False}
    if not isinstance(value, str):
        raise ValueError('A source comment must be text or unknown')
    if len(value) > MAX_COMMENT_CHARS:
        raise DeviceThermalResourceError('A selected source comment exceeds65536 characters')
    digest = sha256()
    for start in range(0, len(value), 8192):
        digest.update(value[start:start + 8192].encode('utf-8'))
    return {'source_comment': value if len(value) <= 2000 else None,
            'source_comment_preview': value[:512], 'source_comment_sha256': digest.hexdigest(),
            'source_comment_truncated': len(value) > 2000}


def _repository_identity(value):
    if value is None:
        return None
    allowed = {'kind', 'sha256', 'byte_length', 'name', 'format', 'scope', 'locator'}
    if not isinstance(value, dict) or len(value) > len(allowed) or set(value) - allowed:
        raise ValueError('Repository identity must contain bounded scalar metadata only')
    for _key, item in value.items():
        if isinstance(item, str):
            _text(item, 'Repository identity')
        elif item is not None and (type(item) is not int or item < 0 or item > 2**63 - 1):
            raise ValueError('Repository identity values must be bounded text or nonnegative integer metadata')
    digest = value.get('sha256')
    if digest is not None and (not isinstance(digest, str) or len(digest) != 64
                               or any(char not in '0123456789abcdefABCDEF' for char in digest)):
        raise ValueError('A repository SHA-256 must have64 hexadecimal characters')
    return dict(value)


def _bounded_output(value):
    encoded = json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(',', ':')).encode()
    if len(encoded) > MAX_OUTPUT_BYTES:
        raise DeviceThermalResourceError('Native-device thermal output exceeds4MiB')
    return value


def _graph(record, family):
    graph = getattr(record, 'graph_v_i' if family == 'channel' else 'graph_i_e', None)
    if isinstance(graph, (list, tuple)):
        if len(graph) != 2 or any(not hasattr(row, '__len__') or len(row) > MAX_GRAPH_POINTS for row in graph):
            raise DeviceThermalResourceError('Each selected graph supports2 rows and at most10000 points')
    if graph is None:
        raise ValueError('The selected record has no required current graph')
    graph = np.asarray(graph)
    if graph.ndim != 2 or graph.shape[0] != 2:
        raise ValueError('Selected current curves require a paired2-row graph')
    if graph.shape[1] > MAX_GRAPH_POINTS:
        raise DeviceThermalResourceError('Each selected graph supports at most10000 points')
    if graph.shape[1] < 2 or graph.dtype.kind not in 'iuf' or not np.all(np.isfinite(graph)):
        raise ValueError('Selected graphs require finite numeric paired samples')
    return graph


def _project(graph):
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        result = np.asarray(graph, dtype=float)
    if not np.all(np.isfinite(result)) or np.any((graph != 0) & (result == 0)):
        raise ValueError('A selected curve exceeds the nonzero finite binary64 sample range')
    return result


def _at(table, query):
    value = table.evaluate(query, dict(table.conditions))
    if value == 0:
        position = int(np.searchsorted(table.x, query))
        exact = position < len(table.x) and table.x[position] == query
        if not exact and 0 < position < len(table.x) and (table.y[position - 1] > 0 or table.y[position] > 0):
            raise ValueError('Positive curve interpolation underflows the supported SI range')
    return value


def _product(values, label):
    result = np.longdouble(1)
    nonzero = True
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        for value in values:
            nonzero = nonzero and value != 0
            result *= np.longdouble(value)
        output = float(result)
    if not math.isfinite(output) or (nonzero and output == 0):
        raise ValueError(f'{label} exceeds the nonzero finite SI range')
    return output


def _sum(values):
    with np.errstate(over='ignore', invalid='ignore'):
        result = float(sum((np.longdouble(value) for value in values), np.longdouble(0)))
    if not math.isfinite(result):
        raise ValueError('Total native-device losses exceed the finite SI range')
    return result


def build_device_loss_table(transistor, request, *, repository_source=None):
    """Build bounded current-then-temperature losses without filesystem access.

    The dependency digest covers selected physics only. It is not a raw file hash.
    """
    if not isinstance(request, dict) or len(request) > len(_FIELDS) or set(request) - _FIELDS:
        raise ValueError('Supply only the declared native-device thermal fields')
    for family in _FAMILIES:
        selection = request.get(family)
        if selection is None:
            continue
        if not isinstance(selection, dict) or len(selection) != 2 or set(selection) != {'indices', 'conditions'}:
            raise ValueError('Family selections require only indices and conditions')
        indices, conditions = selection['indices'], selection['conditions']
        if not isinstance(indices, (list, tuple)) or len(indices) > MAX_RECORDS:
            raise DeviceThermalResourceError('Each source selection supports at most32 records')
        if any(type(index) is not int or index < 0 for index in indices):
            raise ValueError('Source indices must be nonnegative integers')
        allowed = {'v_g'} if family == 'channel' else set(_ENERGY_NUMERIC + _ENERGY_TEXT)
        if not isinstance(conditions, dict) or len(conditions) > len(allowed) or set(conditions) - allowed:
            raise ValueError('Use bounded named source conditions only')
        for name, value in conditions.items():
            if value is not None:
                _text(value, name) if name in _ENERGY_TEXT else _number(value, name)
    if request.get('selected') is not True or request.get('switching_policy') != 'held-current-pwm':
        raise ValueError('Select the held-current PWM device estimate explicitly')
    source = _text(request['source'], 'Thermal path source')
    device_name = _text(transistor.metadata.name, 'Device name')
    repository_source = _repository_identity(repository_source)
    current = _number(request['current_a'], 'Held switch current', minimum=0)
    duty = _number(request['duty'], 'Conduction duty', minimum=0)
    frequency = _number(request['frequency_hz'], 'PWM frequency', minimum=0)
    if duty > 1:
        raise ValueError('Conduction duty must lie in[0,1]')
    switching = frequency > 0 and 0 < duty < 1
    active = {'channel': current > 0 and duty > 0, 'turn_on': switching, 'turn_off': switching}
    if not any(active.values()):
        raise ValueError('No active curve family establishes a temperature domain; use explicit zero-heat steady analysis')
    selected, total_points = {}, 0
    for family, attribute in _FAMILIES.items():
        if not active[family]:
            continue
        selection = request.get(family)
        if not isinstance(selection, dict) or len(selection) != 2 or set(selection) != {'indices', 'conditions'}:
            raise ValueError(f'Select explicit {family} record indices and conditions')
        indices = selection['indices']
        if not isinstance(indices, (list, tuple)) or len(indices) < 2:
            raise ValueError(f'{family} requires at least two selected temperatures')
        if len(indices) > MAX_RECORDS:
            raise DeviceThermalResourceError('Each active family supports at most32 selected records')
        if any(type(index) is not int or index < 0 for index in indices) or len(set(indices)) != len(indices):
            raise ValueError('Source indices must be distinct nonnegative integers')
        names = ('v_g',) if family == 'channel' else ('v_g', 'v_g_off', 'r_g', 'v_supply')
        requested = (_conditions(selection['conditions'], names) if family == 'channel'
                     else _energy_selection(selection['conditions']))
        records = getattr(transistor.switch, attribute)
        rows = []
        for index in indices:
            if index >= len(records):
                raise ValueError(f'{family} source index is outside this device inventory')
            record = records[index]
            temperature = _number(record.t_j, 'Recorded junction temperature')
            recorded = (_conditions({name: getattr(record, name) for name in names}, names)
                        if family == 'channel' else _energy_context(record))
            _match(recorded, requested)
            _comment(record)
            single = family != 'channel' and record.dataset_type == 'single'
            if family != 'channel' and not single and record.dataset_type != 'graph_i_e':
                raise ValueError('Select current-energy curves or exact-current single measurements')
            graph = None if single else _graph(record, family)
            total_points += 1 if single else graph.shape[1]
            if total_points > MAX_SELECTED_POINTS:
                raise DeviceThermalResourceError('Selected curve evidence exceeds1000000 points')
            rows.append((temperature, index, record, recorded, graph))
        for index, row in enumerate(rows):
            for other in rows[:index]:
                _match(row[3], other[3])
        rows.sort(key=lambda row: row[0])
        if any(a[0] == b[0] for a, b in zip(rows, rows[1:], strict=False)):
            raise ValueError('Selected temperature slices must be unique; duplicate rows are ambiguous')
        selected[family] = rows
    dependencies, proof, component_tables, current_bounds = [], [], {}, []
    for family, rows in selected.items():
        temperatures, powers = [], []
        for temperature, index, record, recorded, graph in rows:
            key = f'{family}:{index}'
            if graph is None:
                ix = _number(record.i_x, 'Single measurement current', minimum=0)
                value = _number(record.e_x, 'Single measurement energy', minimum=0)
                if current != ix:
                    raise ValueError('A single measurement requires its exact recorded current')
                domain = [ix, ix]
                dependency = {'representation': 'single', 'current_a': ix, 'energy_j': value}
                bracket = [0, 0]
                sorted_bracket = [0, 0]
                bracket_current, bracket_values = [ix, ix], [value, value]
            else:
                projected = _project(graph)
                x, y = (projected[1], projected[0]) if family == 'channel' else projected
                table = MeasuredTable('channel' if family == 'channel' else 'energy', x, y, {}, key)
                value = _at(table, current)
                domain = [float(table.x[0]), float(table.x[-1])]
                position = int(np.searchsorted(table.x, current))
                sorted_bracket = [position, position] if position < len(table.x) and table.x[position] == current else [position - 1, position]
                bracket_current = [float(table.x[i]) for i in sorted_bracket]
                bracket_values = [float(table.y[i]) for i in sorted_bracket]
                bracket = [int(np.flatnonzero(x == table.x[i])[0]) for i in sorted_bracket]
                dependency = {'representation': 'graph_v_i' if family == 'channel' else 'graph_i_e', 'graph': graph}
            power = _product((current, duty, value), 'Conduction power') if family == 'channel' else _product((frequency, value), 'Switching power')
            dependencies.append({'family': family, 'index': index, 'temperature_c': temperature,
                                 'conditions': recorded, **dependency})
            proof.append({'record_key': key, 'family': family, 'index': index, 'temperature_c': temperature,
                          'recorded_conditions': recorded, 'current_domain_a': domain,
                          'current_sample_bracket': bracket, 'current_sorted_bracket': sorted_bracket,
                          'current_bracket_a': bracket_current, 'current_bracket_values': bracket_values,
                          'current_bracket_basis': 'Original source graph indices; explicit coordinates use validated binary64 samples; single records use point0',
                          'evaluated_drop_v' if family == 'channel' else 'evaluated_energy_j': value,
                          'power_w': power, **_comment(record)})
            current_bounds.append(domain)
            temperatures.append(temperature)
            powers.append(power)
        component_tables[family] = MeasuredTable('energy-condition', temperatures, powers, {}, family + ' derived native loss')
    lower = max(float(table.x[0]) for table in component_tables.values())
    upper = min(float(table.x[-1]) for table in component_tables.values())
    if lower >= upper:
        raise ValueError('Active curve families require a nonzero common temperature interval')
    knots = sorted({lower, upper, *(float(t) for table in component_tables.values() for t in table.x if lower <= t <= upper)})
    values = {family: [_at(component_tables[family], t) if active[family] else 0.0 for t in knots] for family in _FAMILIES}
    total = [_sum(values[family][i] for family in _FAMILIES) for i in range(len(knots))]
    encoded = _encode({'method': 'native-fixed-current-temperature-table-v1', 'records': dependencies})
    digest = sha256(json.dumps(encoded, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return _bounded_output({'method': 'native-fixed-current-temperature-table-v1',
            'device_name': device_name, 'scope': 'One switch junction only; no diode, package or shared-junction heat inferred',
            'temperature_c': knots, 'component_power_w': values, 'total_power_w': total,
            'temperature_domain_c': [lower, upper],
            'current_domain_a': [max(row[0] for row in current_bounds), min(row[1] for row in current_bounds)],
            'active_contributions': active, 'record_evidence': proof,
            'source_identity': {'selected_dependency_sha256': digest,
                'selected_dependency_scope': 'Selected representation, indices, graphs and recorded physics conditions only; no load-time metadata',
                'repository_source': deepcopy(repository_source)},
            'loss_model': {'mode': 'recorded', 'source': source, 'conditions': {},
                'temperature_min_c': lower, 'temperature_max_c': upper, 'temperature_c': knots, 'power_w': total},
            'assumptions': ['Held nonnegative per-switch current; conduction duty is explicitly supplied',
                'Selected held-current PWM model has one on/off pair per period only for0<duty<1; endpoint duty has no events',
                'Zero frequency disables switching; active zero-current switching still evaluates E(0)',
                'Bounded current interpolation precedes independent bounded temperature interpolation; no nearest record or extrapolation'],
            'resource_use': {'selected_records': len(proof), 'selected_points': total_points, 'temperature_knots': len(knots)}})


def analyze_device_electrothermal(transistor, request, *, repository_source=None):
    """Feed one derived recorded table into the existing bounded thermal solver."""
    table = build_device_loss_table(transistor, request, repository_source=repository_source)
    solver_request = {'source': request['source'], 'sink_temperature_c': _number(request['sink_temperature_c'], 'Fixed sink temperature'),
        'common_resistance_k_w': 0, 'branch_resistances_k_w': [_number(request['resistance_k_w'], 'Junction-to-fixed-sink path', minimum=0)],
        'initial_temperatures_c': [_number(request['initial_temperature_c'], 'Initial junction temperature')],
        'loss_models': [table['loss_model']], 'tolerance_k': request.get('tolerance_k', 1e-8),
        'max_iterations': request.get('max_iterations', 200), 'relaxation': request.get('relaxation', 1)}
    solution = electrothermal(solver_request)
    total_table = MeasuredTable('energy-condition', table['temperature_c'], table['total_power_w'], {}, 'Derived total native loss')
    for row in solution['history']:
        if row['power_w'] is not None and row['power_w'][0] == 0:
            _at(total_table, row['temperature_c'][0])
    accepted = solution['accepted_solution']
    breakdown = None
    if accepted is not None:
        temperature = accepted['temperature_c'][0]
        breakdown = {family: _at(MeasuredTable('energy-condition', table['temperature_c'], values, {}, family + ' derived loss'), temperature)
                     for family, values in table['component_power_w'].items()}
        breakdown['sum_component_power_w'] = _sum(breakdown.values())
        breakdown['solver_power_w'] = accepted['power_w'][0]
    return _bounded_output({'operation': 'native-device-electrothermal', 'request': deepcopy(request),
            'table': table, 'solver_request': solver_request, 'result': solution,
            'accepted_component_power_w': breakdown,
            'units': 'A, V, Hz, W, K/W; temperatures C and temperature differences K'})
