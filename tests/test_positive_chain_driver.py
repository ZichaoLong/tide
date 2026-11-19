"""A successful numeric child must not hide host fallback from the device gate."""
import json
import sys

import pytest

from scripts import qualify_positive_chain as driver


@pytest.mark.parametrize("where", ["parent", "restore", "missing_parent", "none"])
def test_device_gate_reads_parent_and_restored_cpp_logs(tmp_path, monkeypatch, where):
    output = tmp_path / "qualification"
    output.mkdir()
    (output / "result.json").write_text(json.dumps(dict(state="within_threshold")))
    if where != "missing_parent":
        (tmp_path / "stderr.log").write_text("")
    warning = "operator aten::frexp.Tensor_out will fall back to run on the CPU\n"
    if where == "parent":
        (tmp_path / "stderr.log").write_text(warning)
    elif where == "restore":
        (output / "restore.log").write_text(warning)
    monkeypatch.setattr(driver.qualification, "main", lambda: 0)
    monkeypatch.setattr(driver.qualification, "configurations", driver.qualification.configurations)
    monkeypatch.setattr(sys, "argv", ["qualify", "--device", "npu:0", "--output-dir", str(output)])
    if where == "none":
        assert driver.main() == 0
    else:
        with pytest.raises(RuntimeError, match="NPU placement gate"):
            driver.main()
    result = json.loads((output / "result.json").read_text())
    assert result["state"] == ("within_threshold" if where == "none" else "failed")
