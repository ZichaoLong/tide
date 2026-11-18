"""Training-boundary, restoration and non-additive diagnostic witnesses."""
import json

import pytest
import torch

from tide import ModelConfig, TideModel, TopologyConfig
from benchmarks.baselines import TransformerBaseline
from benchmarks.costs import TrainingAnchor, main, semantic_work, state_hash, training_step
from benchmarks.stages import SavedAudit, ShapeAudit, StageTimes, stage_hooks


def test_anchor_repeats_exact_optimizer_update_without_mutating_cpu_snapshot():
    torch.manual_seed(751)
    device = torch.device("cpu")
    model = TransformerBaseline(32, 1, window=3, chunk=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=True)
    inputs = torch.randn(2, 5, 32)
    training_step(model, inputs, optimizer, device, "fp32")
    anchor = TrainingAnchor(model, optimizer, device)
    hashes = []
    for _ in range(3):
        assert anchor.restore(model, optimizer) == anchor.identity
        training_step(model, inputs, optimizer, device, "fp32")
        hashes.append((state_hash(model.state_dict()), state_hash(optimizer.state_dict())))
    assert hashes[0] == hashes[1] == hashes[2]
    assert state_hash(anchor.optimizer) == anchor.identity["optimizer_sha256"]


def test_nested_stage_times_partition_only_intrusive_observation():
    ticks = iter([0., 1., 3., 5.])
    syncs = []
    observer = StageTimes(torch.device("cpu"), clock=lambda: next(ticks), sync=syncs.append)
    with observer.scope("parent"):
        with observer.scope("child"):
            pass
    rows = {r["path"]: r for r in observer.report()["rows"]}
    assert rows["parent"]["inclusive_seconds"] == 5
    assert rows["parent"]["exclusive_seconds"] == 3
    assert rows["parent/child"]["exclusive_seconds"] == 2
    assert len(syncs) == 4


@pytest.mark.parametrize("kind", ["stages", "runtime", "saved"])
def test_hooks_cover_batched_timed_flush_without_changing_update(kind):
    torch.manual_seed(193)
    device = torch.device("cpu")
    config = ModelConfig(topology=TopologyConfig(kind="multiscale", n=2, hubs=1,
                         profile="compact-5-dd4", timing="timed"), d=64, memory="X",
                         sharing="region_shared", region_batch_size=4)
    model = TideModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=True)
    inputs = torch.randn(1, 2, 64)
    anchor = TrainingAnchor(model, optimizer, device)
    reference_loss = training_step(model, inputs, optimizer, device, "fp32")
    expected = state_hash(model.state_dict()), state_hash(optimizer.state_dict())
    anchor.restore(model, optimizer)
    observer = StageTimes(device) if kind == "stages" else ShapeAudit() if kind == "runtime" else SavedAudit(model)
    from contextlib import nullcontext
    context = (observer if kind == "runtime" else torch.autograd.graph.saved_tensors_hooks(observer.pack, observer.unpack)
               if kind == "saved" else nullcontext())
    with stage_hooks(model, observer), context:
        actual_loss = training_step(model, inputs, optimizer, device, "fp32", observer)
    assert actual_loss == reference_loss
    assert (state_hash(model.state_dict()), state_hash(optimizer.state_dict())) == expected
    report = observer.report()
    paths = [r["path"] for r in report["operators" if kind == "runtime" else "rows"]]
    assert any("training/flush/regions.evaluate" in p for p in paths)
    assert any("training/forward/regions.evaluate/regional.scoring" in p for p in paths)
    if kind == "saved":
        assert report["total"]["peak_live_unique_storage_bytes"] > 0
        assert report["total"]["current_live_unique_storage_bytes"] == 0
    # Class and global function aliases must be restored even after failure.
    with pytest.raises(RuntimeError, match="intentional"):
        with stage_hooks(model, observer):
            raise RuntimeError("intentional")
    anchor.restore(model, optimizer)
    assert training_step(model, inputs, optimizer, device, "fp32") == reference_loss


def test_transformer_work_distinguishes_useful_pairs_and_masked_tile_capacity():
    model = TransformerBaseline(32, 2, window=3, chunk=2)
    report = semantic_work(model, torch.zeros(1, 5, 32), torch.device("cpu"), "fp32")
    # 1+2+3+3+3 legal causal pairs per layer; tile rectangles 2*2+2*4+1*3.
    assert report["effective_attention_pairs"] == 24
    assert report["attention_operand_rectangle_pairs"] == 30
    assert report["masked_rectangle_pairs"] == 6


def test_cost_cli_keeps_normal_and_restored_trace_separate(tmp_path):
    config = ModelConfig(topology=TopologyConfig(kind="fc", n=2, layers=1), d=64)
    cfg, output = tmp_path/"config.json", tmp_path/"costs.json"
    cfg.write_text(json.dumps(config.to_dict()))
    assert main(["--config", str(cfg), "--output", str(output), "--subjects", "transformer_D",
                 "--batch", "1", "--length", "3", "--warmup", "1", "--steps", "2",
                 "--transformer-candidates", "standard", "chunk256", "contiguous_qkv",
                 "--diagnostics", "runtime", "trace"]) == 0
    record = json.loads(output.read_text())
    assert len(record["observations"]) == 3
    assert len({r["initial_parameter_sha256"] for r in record["observations"]}) == 1
    assert len({r["input_sha256"] for r in record["observations"]}) == 1
    assert all(r["anchor"] == record["observations"][0]["anchor"] for r in record["observations"])
    row = record["observations"][0]
    assert len(row["normal"]["seconds"]) == 2
    assert row["normal"]["start_identity"] == row["diagnostics"]["runtime"]["start_identity"] == row["diagnostics"]["trace"]["start_identity"]
    assert row["normal"]["losses"][0] == row["diagnostics"]["runtime"]["loss"] == row["diagnostics"]["trace"]["loss"]
    assert row["diagnostics"]["trace"]["summary"]["complete_event_count"] > 0
    assert all(r["parameter_target_kind"] in {"total", "effective_touched"} for r in record["joint_matches"].values())
    from benchmarks.cost_report import main as report_main
    report = tmp_path/"report.md"
    assert report_main([str(output), "--output", str(report), "--limit", "3"]) == 0
    assert "transformer_D/chunk256" in report.read_text()


def test_failed_diagnostic_preserves_normal_samples_and_reason(tmp_path, monkeypatch):
    import benchmarks.costs as costs
    def fail(*args, **kwargs):
        raise RuntimeError("retained profiler failure")
    monkeypatch.setattr(costs, "trace_step", fail)
    config = ModelConfig(topology=TopologyConfig(kind="fc", n=2, layers=1), d=64)
    cfg, output = tmp_path/"config.json", tmp_path/"costs.json"
    cfg.write_text(json.dumps(config.to_dict()))
    assert main(["--config", str(cfg), "--output", str(output), "--subjects", "transformer_C",
                 "--length", "2", "--warmup", "0", "--steps", "1", "--diagnostics", "trace"]) == 1
    record = json.loads(output.read_text())
    row = record["observations"][0]
    assert record["state"] == row["state"] == row["diagnostics"]["trace"]["state"] == "failed"
    assert row["normal"]["state"] == "completed_finite" and len(row["normal"]["seconds"]) == 1
    assert "retained profiler failure" in row["diagnostics"]["trace"]["error"]
    from benchmarks.cost_report import main as report_main
    report = tmp_path/"report.md"
    assert report_main([str(output), "--output", str(report)]) == 0
    assert "retained profiler failure" in report.read_text()
