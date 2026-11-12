"""Read-budget regressions for the reused source snapshot boundary."""
from io import BytesIO

import pytest

from transistordatabase.gui_web.backend.device_electrothermal_routes import _snapshot


@pytest.mark.parametrize('allowance', [-2, -1, 1.5, True, False, None])
def test_invalid_read_allowance_is_rejected_before_source_access(allowance):
    """A negative read size must never reach stream.read and consume all bytes."""
    class UntouchedRepository:
        def _storage_path(self, name):
            raise AssertionError('Source access occurred before allowance validation')

    with pytest.raises(ValueError, match='nonnegative integer'):
        _snapshot(UntouchedRepository(), 'synthetic', max_bytes=allowance)


@pytest.mark.parametrize('allowance', [0, 3])
def test_small_read_allowance_reads_only_its_overflow_sentinel(allowance):
    """Even a zero allowance reads only the bounded sentinel, then refuses decode."""
    reads = []

    class RecordingStream(BytesIO):
        def read(self, size=-1):
            reads.append(size)
            return super().read(size)

    class SourcePath:
        def open(self, mode):
            assert mode == 'rb'
            return RecordingStream(b'{"not":"a device"}')

    class Repository:
        def _storage_path(self, name):
            return SourcePath()

    with pytest.raises(ValueError, match='bounded read allowance'):
        _snapshot(Repository(), 'synthetic', max_bytes=allowance)
    assert reads == [allowance + 1]
