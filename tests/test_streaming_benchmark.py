import json
from pathlib import Path
import subprocess
import sys
import pytest
import torch


def binary():
    import _tide_native
    return Path(_tide_native.__file__).with_name("tidegraph-streaming-bench")


def literal_checks(dtype, width=3, batch=2, rings=2, ticks=17, seed=7):
    states, incoming, outputs = {}, {}, []
    for ring in range(rings):
        for b in range(batch):
            incoming[b, ring] = .1+torch.sin(torch.arange(width, dtype=dtype)*.13+b*.17+ring*.19+(seed%997)*.01)*.03
    retention = torch.sigmoid(torch.full((width,), -.3, dtype=dtype))
    for t in range(ticks):
        for ring in range(rings):
            for b in range(batch):
                owner = (b, ring, t%8)
                h = incoming[b, ring]
                proposal = retention*states.get(owner, torch.zeros(width, dtype=dtype))+h
                full = h+torch.tanh(proposal*.1+.01)
                states[owner] = proposal
                incoming[b, ring] = full*.2+full*.2
                if t%8 == 7:
                    outputs.append(full)
    return {"check/output_checksum": sum(x.double().sum().item() for x in outputs),
            "check/state_checksum": sum(x.double().sum().item() for x in states.values()),
            "check/pending_checksum": sum(x.double().sum().item() for x in incoming.values())}


@pytest.mark.parametrize("api,workers,packed", [("functional", 1, 0), ("cursor", 1, 0),
                                               ("functional", 3, 1), ("cursor", 3, 1)])
def test_native_benchmark_has_an_independent_literal_anchor(dtype, api, workers, packed, tmp_path):
    out = tmp_path/"native"
    command = [str(binary()), "--device", "cpu", "--dtype", str(dtype).split('.')[-1],
               "--run-id", "literal", "--output-dir", str(out), "--nodes", "64", "--active-rings", "2",
               "--batch", "2", "--width", "3", "--ticks", "17", "--seed", "7", "--api", api,
               "--workers", str(workers), "--packed", str(packed), "--warmup", "1", "--repetitions", "2"]
    subprocess.run(command, capture_output=True, text=True, check=True)
    events = [json.loads(line) for line in (out/"metrics.jsonl").read_text().splitlines()]
    assert len(events) == 2
    expected = literal_checks(dtype)
    for step, event in enumerate(events):
        assert event["step"] == event["sequence"] == step and event["run_id"] == "literal"
        actual = event["metrics"]
        for key, value in expected.items():
            assert actual[key] == pytest.approx(value, abs=1e-10 if dtype == torch.float64 else 1e-6,
                                                rel=1e-8 if dtype == torch.float64 else 1e-5)
        assert actual["work/candidate_events"] == 68 and actual["work/visited_edges"] == 136
        assert actual["work/update_calls"] == (34 if packed else 68)
        assert actual["work/cached_states"] == 32 and actual["work/pending_messages"] == 8
        assert actual["perf/advance_seconds"] > 0 and actual["memory/graph_identity_bytes"] > 0
        assert (actual["perf/snapshot_seconds"] > 0) == (api == "cursor")
    before = (out/"metrics.jsonl").read_bytes()
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert (out/"metrics.jsonl").read_bytes() == before


@pytest.mark.parametrize("arguments", [[], ["--device", "npu"], ["--device", "cpu", "--nodes", "9"],
                                       ["--device", "cpu", "--packed", "2"], ["--device", "cpu", "--ticks", "3x"],
                                       ["--device", "cpu", "--dtype", "float16"]])
def test_benchmark_invalid_request_does_not_create_output(arguments, tmp_path):
    out = tmp_path/"invalid"
    result = subprocess.run([str(binary()), "--run-id", "invalid", "--output-dir", str(out), *arguments], capture_output=True)
    assert result.returncode != 0 and not out.exists()


def test_benchmark_record_matches_the_native_events(tmp_path):
    out = tmp_path/"record"
    script = Path(__file__).resolve().parents[1]/"scripts/benchmark_streaming.py"
    result = subprocess.run([sys.executable, str(script), "--device", "cpu", "--build-dir", str(binary().parent),
                            "--allow-dirty", "--tracking", "off", "--output-dir", str(out),
                            "--nodes", "16", "--active-rings", "2", "--batch", "1", "--width", "3",
                            "--ticks", "8", "--warmup", "0", "--repetitions", "2"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr+result.stdout
    manifest = json.loads((out/"run.json").read_text()); summary = json.loads((out/"summary.json").read_text())
    assert manifest["status"] == summary["status"] == "completed"
    assert manifest["run_id"] == summary["run_id"] and summary["native_exit_code"] == 0
    assert summary["observations"] == 2 and summary["tracking"]["status"] == "disabled"
    assert (out/"metrics.jsonl").read_bytes() == (out/"native/metrics.jsonl").read_bytes()
