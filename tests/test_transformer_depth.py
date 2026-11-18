"""Supplementary depth comparisons retain real cache and workload identities."""
import json

import pytest
import torch

from benchmarks import transformer_depth
from benchmarks.baselines import TransformerBaseline, parameter_count
from benchmarks.matching import match_transformer
from tide import TideModel


@pytest.mark.parametrize("parameters", [8778358224, 16707220176])
@pytest.mark.parametrize("layers", [32, 48])
def test_large_matches_count_real_model_parameters_without_allocating_payloads(parameters, layers):
    match = match_transformer(layers, parameters)
    with torch.device("meta"):
        model = TransformerBaseline(**match["constructor"])
    assert len(model.blocks) == layers
    assert parameter_count(model) == match["actual"]["parameters"]
    assert abs(parameter_count(model)/parameters-1) < .001


def test_supplementary_cli_uses_real_cache_without_constructing_tide(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("supplementary comparison must not allocate Tide")
    monkeypatch.setattr(TideModel, "__init__", forbidden)
    path = tmp_path/"depth.json"
    args = ["--parameters", "33000", "--layers", "2", "--batch", "2", "--window", "4",
            "--prefill", "5", "--prefill-chunk", "2", "--decode", "3", "--samples", "2",
            "--warmup", "1", "--threads", "2", "--devices", "cpu:0,cpu:1", "--output", str(path)]
    assert transformer_depth.main(args) == 0
    record = json.loads(path.read_text())
    assert record["state"] == "completed"
    assert not record["comparison"]["replaces_joint_classes"]
    assert record["diagnostics"]["finite_outputs"] and record["diagnostics"]["finite_private_state"]
    for observed in record["observations"]:
        assert [r["state_before"] for r in observed["calls"]] == [0, 2, 4, 5, 6, 7]
        assert observed["input_positions"] == 16 and observed["final_state_position"] == 8
        assert observed["stable_decode"]["input_positions"] == 4
        assert observed["stable_decode"]["calls"] == 2
        memory = observed["state_memory"]
        expected_bytes = 2*2*2*3*record["spec"]["d"]*4
        assert memory["payload_bytes"] == expected_bytes
    original = path.read_bytes()
    with pytest.raises(SystemExit):
        transformer_depth.main(args)
    assert path.read_bytes() == original
