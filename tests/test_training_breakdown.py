"""Continuing updates and restored observers share a real nonempty anchor."""
import json

import pytest
import torch

from benchmarks.training_breakdown import copy_tree, identity_tree, main
from benchmarks.costs import state_hash
from tide import ModelConfig, TopologyConfig


@pytest.mark.parametrize("subject,kind", [("tide", "fc"), ("transformer", "fc"), ("tide", "hb")])
def test_breakdown_replays_nonempty_training_anchor(tmp_path, subject, kind):
    cfg = (ModelConfig(topology=TopologyConfig(kind="fc", n=2, layers=1), d=64, kv_width=32)
           if kind == "fc" else ModelConfig(topology=TopologyConfig(kind="multiscale", n=2, hubs=1,
               profile="compact-5-dd4", timing="settle"), d=64, memory="X", kv_width=32,
               checkpoint_reception=True, region_batch_size=4, ssd_summary_backend="prefix", ssd_scan_backend="doubling"))
    length = 3 if kind == "fc" else 8
    config, output = tmp_path/"config.json", tmp_path/"result.json"
    config.write_text(json.dumps(cfg.to_dict()))
    assert main(["--config", str(config), "--subject", subject, "--transformer-layers", "2",
                 "--batch", "1", "--length", str(length), "--chunk", "2", "--warmup", "1",
                 "--samples", "2", "--threads", "1", "--output", str(output)]) == 0
    result = json.loads(output.read_text())
    assert result["state"] == "completed_finite"
    assert result["anchor"]["private_state_position"] == length
    assert result["optimizer"]["foreach"] is True
    assert [r["state_after"] for r in result["normal"]] == [2*length, 3*length]
    assert [x["continuation"]["optimizer_step_min"] for x in result["normal"]] == [2, 3]
    for row in result["diagnostics"].values():
        assert row["state"] == "completed_finite"
        assert row["private_state_start_sha256"] == result["anchor"]["private_state_sha256"]
        assert row["loss"] == result["normal"][0]["loss"]
        assert row["update_identity"] == result["first_measured_update"]
        assert all(v["exact"] for v in row["normal_update_comparison"].values())
    with pytest.raises(SystemExit):
        main(["--config", str(config), "--output", str(output)])


def test_private_state_copy_and_hash_include_values_beyond_repr():
    # Large enough to abbreviate repr; changes inside the omitted part must count.
    value = {"times": torch.arange(2000, dtype=torch.int64), "state": torch.zeros(2000)}
    copied = copy_tree(value, "cpu")
    original_hash = state_hash(identity_tree(value))
    copied["times"][800] += 1
    assert state_hash(identity_tree(copied)) != original_hash
    assert value["times"][800] == 800
