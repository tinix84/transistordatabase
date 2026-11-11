"""Native core persistence without legacy identity guesses or unsafe reconstruction."""
from copy import deepcopy
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path

import numpy as np
import pytest

from transistordatabase.core.models import (
    ChannelCharacteristics, ElectricalRatings, LinearizedModel, RawMeasurementData,
    ThermalProperties, Transistor, TransistorMetadata,
)
from transistordatabase.core.repository import JsonTransistorLoader
from transistordatabase.core.serialization import VERSION, core_from_json_document, core_to_json_document


def native_fixture():
    """Declare an unlisted actual vendor and unknown housing without coercion."""
    transistor = Transistor(
        TransistorMetadata("Preserved_Device", "MOSFET", "source author", "Unlisted Véndor", "",
            comment="source identity and declarations remain exact",
            datasheet_date=datetime(2020, 1, 2, 3, 4, 5, tzinfo=timezone(timedelta(hours=2)))),
        ElectricalRatings(1200., 75., 40., 175.), ThermalProperties(.0001, .0002))
    transistor.switch.channel_data = [ChannelCharacteristics(25., np.array([[0.,1.],[0.,10.]]), None)]
    transistor.switch.raw_measurement_data = [RawMeasurementData("dpt_u_i", t_j=None,
        dpt_on_id=[np.array([[0.,1.e-9],[0.,2.]])])]
    transistor.switch.metadata = {"kind":"uninterpreted user metadata", "shape":[None, "unknown"],
        "typed": (np.bool_(False), np.uint64(2**63 + 1))}
    transistor.diode.q_rr = 100e-9
    transistor._working_point = {"switch":LinearizedModel(25.,10.,.01,.5,None)}
    return transistor


def test_complete_native_save_reload(tmp_path):
    """Preserve identity, datetimes, None, arrays, recovery and working-point types."""
    transistor = native_fixture()
    path = tmp_path / "native.json"
    loader = JsonTransistorLoader()
    loader.save_to_json(transistor, path)
    restored = loader.load_from_json(path)
    assert restored.metadata == transistor.metadata
    assert restored.metadata.housing_type == ""
    assert restored.switch.channel_data[0].v_g is None
    np.testing.assert_array_equal(restored.switch.raw_measurement_data[0].dpt_on_id[0], transistor.switch.raw_measurement_data[0].dpt_on_id[0])
    assert restored.switch.metadata == transistor.switch.metadata
    assert restored.diode.q_rr == 100e-9
    assert isinstance(restored.working_point["switch"], LinearizedModel)


@pytest.mark.parametrize("dtype", ["float64", "longdouble", "int64", "bool"])
def test_array_dtype_and_empty_shape_preserved(dtype):
    """Keep numeric dtype/empty shape and extended precision without float narrowing."""
    transistor = native_fixture()
    array = np.array([[1, 0], [0, 1]], dtype=dtype)
    if dtype == "longdouble":
        array[0, 0] = np.longdouble("1.000000000000000001")
    transistor.switch.metadata = {"array":array, "empty":np.empty((2,0),dtype=dtype)}
    document = json.loads(json.dumps(core_to_json_document(transistor), allow_nan=False))
    restored = core_from_json_document(document)
    assert restored.switch.metadata["array"].dtype == array.dtype
    np.testing.assert_array_equal(restored.switch.metadata["array"], array)
    assert restored.switch.metadata["empty"].shape == (2,0)


@pytest.mark.parametrize("version", [VERSION + 1, True, "1"])
def test_unknown_native_version_fails_before_legacy(tmp_path, version):
    """Never feed incompatible native archives into the legacy identity validator."""
    document = core_to_json_document(native_fixture())
    document["version"] = version
    path = tmp_path / "unsupported-version.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="schema/version"):
        JsonTransistorLoader().load_from_json(path)


def test_no_dynamic_class_or_extra_field_reconstruction():
    """Only registered models and exact field sets are reconstructible."""
    document = core_to_json_document(native_fixture())
    arbitrary = deepcopy(document)
    arbitrary["transistor"]["name"] = "builtins.eval"
    with pytest.raises(ValueError, match="Unknown core aggregate"):
        core_from_json_document(arbitrary)
    document["transistor"]["fields"]["surprise"] = "not silently dropped"
    with pytest.raises(ValueError, match="fields"):
        core_from_json_document(document)


def test_version_one_raw_record_receives_only_additive_measurement_defaults():
    """Load the old native field set while preserving its original raw sample arrays."""
    device = native_fixture()
    document = core_to_json_document(device)
    document["version"] = 1
    records = document["transistor"]["fields"]["switch"]["fields"]["raw_measurement_data"]
    raw_fields = records["values"][0]["fields"]
    for key in ("time_s", "voltage_v", "current_a", "gate_v", "evidence"):
        raw_fields.pop(key)
    restored = core_from_json_document(document)
    [raw] = restored.switch.raw_measurement_data
    assert raw.evidence is None and raw.gate_v is None and raw.time_s is None
    np.testing.assert_array_equal(raw.dpt_on_id[0], device.switch.raw_measurement_data[0].dpt_on_id[0])
    assert restored.metadata == device.metadata
    document["version"] = VERSION
    with pytest.raises(ValueError, match="field"):
        core_from_json_document(document)


def test_nonfinite_failure_leaves_new_destination_absent(tmp_path):
    """Validate unsupported values before opening the destination file."""
    transistor = native_fixture()
    transistor.switch.channel_data[0].graph_v_i[0, 0] = np.nan
    path = tmp_path / "invalid.json"
    with pytest.raises(ValueError, match="finite"):
        JsonTransistorLoader().save_to_json(transistor, path)
    assert not path.exists()


def test_legacy_json_load_stays_read_only_and_native_roundtrips(tmp_path):
    """Continue reading actual existing flat legacy data without source writes."""
    source = Path(__file__).parent / "test_data/database/CREE_C3M0016120K.json"
    before = source.read_bytes()
    loader = JsonTransistorLoader()
    original = loader.load_from_json(source)
    destination = tmp_path / "converted-native.json"
    loader.save_to_json(original, destination)
    restored = loader.load_from_json(destination)
    assert restored.metadata == original.metadata
    np.testing.assert_array_equal(restored.switch.channel_data[0].graph_v_i, original.switch.channel_data[0].graph_v_i)
    assert source.read_bytes() == before
