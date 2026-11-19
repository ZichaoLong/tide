"""A real continued update checks replay sufficiency and input identity guards."""
import json
import os

import pytest
import torch

import scripts.replay_saved_training_update as replay
from benchmarks.inference import source_identity
from benchmarks.workloads import tensor_hash
from scripts.check_execution_candidates import extra_anchor
from scripts.check_training_paths import build_candidate, observe, training_data
from scripts.qualify_precision import THRESHOLDS
from tide import ModelConfig, TideModel, TopologyConfig, placement_adamw


@pytest.mark.parametrize("owner", ["baseline", "weight_access"])
def test_saved_state_replays_real_fourth_step_and_rejects_changed_input(tmp_path, monkeypatch, owner):
    config = ModelConfig(topology=TopologyConfig(n=2, layers=1), d=64, memory="K",
        kv_width=32, sharing="independent", ffn_width=128, region_batch_size=4,
        checkpoint_reception=True, attention_backend="math")
    torch.manual_seed(152)
    inputs = [torch.randn(1, 4, 64) for _ in range(4)]
    model = TideModel(config)
    optimizer = placement_adamw(model, lr=1e-4, foreach=False)
    _, state, _ = observe(model, optimizer, None, inputs[0], "fp32", [torch.device("cpu")], 2, update=True)
    anchor = dict(training=training_data(model, optimizer, state), extra=extra_anchor(model, torch.device("cpu")),
                  config=config.to_dict(), inputs=inputs)
    options = dict(ffn_packing="rectangular", ffn_weight_access="grouped" if owner == "weight_access" else "gather")
    model, optimizer, state = build_candidate(config, owner, anchor["training"], [torch.device("cpu")],
                                             "fp32", "auto", overrides=options)
    (tmp_path / owner).mkdir()
    torch.save(anchor, tmp_path / "anchor.pt")
    for index in range(3):
        updated, state, _ = observe(model, optimizer, state, inputs[index + 1], "fp32", [torch.device("cpu")], 2, update=True)
        if index:
            torch.save(updated, tmp_path / owner / f"update-{index}.pt")
    original = dict(device="cpu", precision="fp32", parameters=sum(p.numel() for p in model.parameters()),
        source=source_identity(), tolerance=THRESHOLDS["fp32"], threads=1,
        base_config=config.to_dict(), input_hashes=[tensor_hash(x) for x in inputs],
        resolved_candidates={owner: options}, workload=dict(length=4, chunk=2),
        state="finite_deviation", artifacts=[])
    identities = []
    for relative in ["anchor.pt", owner + "/update-1.pt", owner + "/update-2.pt"]:
        path = tmp_path / relative
        s = path.stat()
        digest = replay.sha(path)
        original["artifacts"].append(dict(path=relative, bytes=s.st_size, sha256=digest))
        identities.append(dict(path=str(path), bytes=s.st_size, manifest_sha256=digest,
                               device=s.st_dev, inode=s.st_ino, mtime_ns=s.st_mtime_ns, content_sha256_verified=True))
    (tmp_path / "result.json").write_text(json.dumps(original))
    verified = dict(state="diagnosis_complete", torch=torch.__version__, input_files=identities,
                    source_record=dict(sha256=replay.sha(tmp_path / "result.json")))
    verified_path = tmp_path / "verified.json"
    verified_path.write_text(json.dumps(verified))
    before_files = {str(p): replay.sha(p) for p in tmp_path.rglob("*") if p.is_file()}
    real_observe = replay.observe
    def stale_gradients(*args, **kwargs):
        for parameter in args[0].parameters():
            parameter.grad = torch.full_like(parameter, 100.)
        return real_observe(*args, **kwargs)
    monkeypatch.setattr(replay, "observe", stale_gradients)
    record = replay.run(tmp_path, verified_path, owner)
    assert record["state"] == "exact" and record["positions"] == [12, 16]
    assert record["comparison"]["comparison"]["exact"] and record["rng_unchanged"]
    assert record["config"]["region_batch_size"] == 4
    assert record["config"]["ffn_weight_access"] == options["ffn_weight_access"]
    assert record["original_state"] == "finite_deviation"
    assert {str(p): replay.sha(p) for p in tmp_path.rglob("*") if p.is_file()} == before_files
    target = tmp_path / owner / "update-1.pt"
    s = target.stat()
    os.utime(target, ns=(s.st_atime_ns, s.st_mtime_ns + 1))
    with pytest.raises(ValueError, match="stat guard changed"):
        replay.run(tmp_path, verified_path, owner)
