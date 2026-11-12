"""Resource bounds for optional switching-loss frequency populations."""
import json
from dataclasses import fields, is_dataclass
from numbers import Real


def _metadata_bound(value, limit=65536):
    """Count actual metadata occurrences before encoding, including shared values."""
    pending = [(value, 0)]
    nodes = 0
    text_bytes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > 2048 or depth > 8:
            raise ValueError("Frequency grid metadata exceeds structural limits")
        if isinstance(item, str):
            if len(item) > 2048:
                raise ValueError("Frequency grid text is limited to 2048 characters per field")
            text_bytes += len(item.encode('utf-8'))
            if text_bytes > limit:
                raise ValueError("Frequency grid metadata exceeds its text budget")
        elif isinstance(item, dict):
            if len(item) > 128:
                raise ValueError("Frequency grid metadata mappings are limited to 128 entries")
            pending.extend((child, depth + 1) for pair in item.items() for child in pair)
        elif isinstance(item, (list, tuple)):
            if len(item) > 128:
                raise ValueError("Frequency grid metadata lists are limited to 128 entries")
            pending.extend((child, depth + 1) for child in item)
        elif is_dataclass(item):
            members = fields(item)
            if len(members) > 128:
                raise ValueError("Frequency grid metadata has too many fields")
            pending.extend((getattr(item, member.name), depth + 1) for member in members)
        elif item is not None and not isinstance(item, (Real, bool)):
            raise ValueError("Frequency grid metadata must contain bounded scalar values")


def validate_grid_sources(devices, catalog=None):
    """Bound graph shapes before casts and metadata before repeated scalar work."""
    total = 0
    seen = set()
    for device in devices:
        if device is None or id(device) in seen:
            continue
        seen.add(id(device))
        families = (device.switch.channel_data, device.diode.channel_data,
                    device.switch.e_on_data, device.switch.e_off_data)
        for records in families:
            if len(records) > 128:
                raise ValueError("Frequency grid sources are limited to 128 records per family")
            for record in records:
                metadata = {}
                for key, value in vars(record).items():
                    if not key.startswith('graph_') or value is None:
                        metadata[key] = value
                        continue
                    shape = getattr(value, 'shape', None)
                    if shape is not None:
                        if len(shape) != 2 or shape[0] != 2:
                            raise ValueError("Frequency grid graphs must have shape 2 by N")
                        count = shape[1]
                    elif isinstance(value, (list, tuple)) and len(value) == 2:
                        if not all(isinstance(row, (list, tuple)) for row in value):
                            raise ValueError("Frequency grid graphs require two bounded rows")
                        count = len(value[0])
                        if len(value[1]) != count:
                            raise ValueError("Frequency grid graph rows must have equal lengths")
                    else:
                        raise ValueError("Frequency grid graphs require bounded arrays or list rows")
                    total += count
                    if count > 10000 or total > 20000:
                        raise ValueError("Frequency grid sources exceed the 20000 sample-coordinate budget")
                    if any(not isinstance(sample, Real) or isinstance(sample, bool)
                           for row in value for sample in row):
                        raise ValueError("Frequency grid samples must be real scalar coordinates")
                _metadata_bound(metadata)
    if catalog is not None:
        _metadata_bound(catalog)
        if _encoded_size(catalog, 65536) > 65536:
            raise ValueError("Frequency grid catalog metadata exceeds 65536 bytes")


def _encoded_size(value, maximum):
    """Stop streamed encoding at the declared byte budget."""
    size = 0
    for chunk in json.JSONEncoder(allow_nan=False).iterencode(value):
        size += len(chunk.encode('utf-8'))
        if size > maximum:
            return size
    return size


def bounded_grid_result(result):
    """Reject oversized results without constructing an entire encoded copy."""
    if _encoded_size(result, 4 * 1024 * 1024) > 4 * 1024 * 1024:
        raise ValueError("Frequency grid result exceeds 4 MiB")
    return result
