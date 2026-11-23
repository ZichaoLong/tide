"""Independent diagnostic formula and immutable failure-evidence admission."""
import json
from types import SimpleNamespace

import pytest

from pilotlab.diagnose_shannon3 import entropy_reference, file_hash, replay


def test_real_failed_row_is_within_fp32_range_and_has_both_senders():
    result = entropy_reference([6.040277004241943, -10.60705280303955], [0., 0.], [1., 1.], 1., 1.)
    assert result["positive_support"] == [True, True]
    assert 5.8e-8 < result["values"][1] < 6e-8
    assert result["values"][0] < 1.
    assert sum(result["values"]) == pytest.approx(1., abs=1e-15)


def test_reference_preserves_real_mass_remaining_after_float32_caps():
    result = entropy_reference([6., 6., -34.], [0., 0., 0.],
                               [0.10000000149011612, 0.8999999761581421, 1.], 1., 1.)
    assert result["values"][2] == pytest.approx(2.2351741790771484e-8, rel=1e-13)


def test_changed_failure_payload_is_rejected_before_runtime_or_output(tmp_path):
    capture = tmp_path / "failure.pt"
    capture.write_bytes(b"initial bytes")
    result = tmp_path / "result.json"
    result.write_text(json.dumps({"failure_capture": {"path": str(capture), "sha256": file_hash(capture)}}))
    capture.write_bytes(b"changed bytes")
    output = tmp_path / "new-output"
    with pytest.raises(ValueError, match="hash mismatch"):
        replay(SimpleNamespace(result=result, output=output))
    assert not output.exists()


def test_existing_diagnostic_evidence_is_never_replaced(tmp_path):
    output = tmp_path / "existing-output"
    output.mkdir()
    marker = output / "report.json"
    marker.write_text("original evidence")
    with pytest.raises(FileExistsError, match="never replaced"):
        replay(SimpleNamespace(result=tmp_path / "unused", output=output))
    assert marker.read_text() == "original evidence"
