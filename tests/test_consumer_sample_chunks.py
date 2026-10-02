"""Physical sample slices preserve the logical loss, continuation and update."""
import json
import os
import subprocess
import sys
import pytest
import torch
from test_online_consumer import ROOT, make_continuous_packet, ranked_graph, native_text
from online_consumer_support import observer, same
from tools.online_bench.host import run


# Cover all graph families, both schedules, memory modules and optimizers without
# multiplying every policy. Batch five deliberately leaves a one-sample tail.
CASES = [
    ("pdg", "add", "streaming", "sgd", True, False, True),
    ("pdg", "attention", "prefill", "adamw", False, True, True),
    ("timed-dag", "add", "prefill", "adamw", False, True, True),
    ("timed-dag", "attention", "streaming", "sgd", False, False, True),
    ("settle", "add", "streaming", "sgd", True, False, True),
    ("settle", "attention", "prefill", "adamw", False, False, True),
    ("pdg", "attention", "streaming", "sgd", False, True, False),
    ("settle", "add", "prefill", "sgd", True, False, False),
]


def compare_records(actual, expected, batch):
    wanted = {(r["step"], r["cut"]): r for r in expected if r["kind"] == "window"}
    covered = {key: [] for key in wanted}
    for row in actual:
        if row["kind"] != "window":
            continue
        row = dict(row)
        first, stop, logical = row.pop("sample_range")
        assert logical == batch and 0 <= first < stop <= batch
        key = row["step"], row["cut"]
        reference = dict(wanted[key])
        for name in ("outputs", "states", "history", "pending", "events", "ledger"):
            reference[name] = [x for x in reference[name] if first <= x[0] < stop]
        # Preserve each range's original stable order, including duplicate edge
        # and pending identities. Do not sort away an ordering error.
        same(row, reference)
        covered[key].extend(range(first, stop))
    assert all(sorted(samples) == list(range(batch)) for samples in covered.values())
    # All parameter gradients (including None) and every parameter generation
    # are checked after one whole-batch update, with both SGD and AdamW.
    same([r for r in actual if r["kind"] != "window"],
         [r for r in expected if r["kind"] != "window"])


def compare(case, implementation, device, dtype, preset, tmp_path):
    family, memory, schedule, optimizer, clear, delayed, training = case
    p = make_continuous_packet(graph=ranked_graph(layers=3, region_width=2, fanout=2, local_span=2, delayed=delayed),
                               memory=memory, width=4, batch=5, tokens=2, vocab=7, clear=clear)
    expected, actual = [], []
    kw = dict(family=family, dtype=dtype, training=training, optimizer=optimizer,
              steps=2, warmup=0 if training else 1, windows_per_step=2, diagnostics=True)
    reference = run(p, implementation="python", device="cpu", schedule="streaming",
                    observer=observer(expected), **kw)
    if implementation == "libtorch":
        binary = os.environ.get("TIDE_ONLINE_BINARY")
        if not binary:
            pytest.skip("standalone consumer not explicitly selected")
        packet = tmp_path/"packet.txt"
        packet.write_text(native_text(p))
        out = tmp_path/"consumer"
        cmd = [binary, "--device="+str(device), "--dtype="+dtype, "--packet="+str(packet), "--output-dir="+str(out),
               "--family="+family, "--preset="+preset, "--schedule="+schedule, "--optimizer="+optimizer,
               "--steps=2", "--warmup="+str(kw["warmup"]), "--windows-per-step=2", "--diagnostics",
               "--sample-chunk-rows=2", "--workers=2", "--packed-sources", "--batch-next"]
        if training:
            cmd.append("--training")
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        assert completed.returncode == 0, completed.stdout+completed.stderr
        actual = [json.loads(line) for line in (out/"diagnostics.jsonl").read_text().splitlines()]
        measured = json.loads((out/"result.json").read_text())
    else:
        native = dict(native_library=os.environ["TIDE_BUILD_DIR"], workers=2,
                      packed_sources=True, batch_next=True) if implementation == "native" else {}
        measured = run(p, implementation=implementation, device=device, schedule=schedule, preset=preset,
                       observer=observer(actual), sample_chunk_rows=2, **native, **kw)
    compare_records(actual, expected, 5)
    for name in ("outputs", "final_cut", "parameters", "input_tokens_per_step"):
        assert measured[name] == reference[name], name
    torch.testing.assert_close(torch.tensor(measured["losses"]), torch.tensor(reference["losses"]), atol=1e-6, rtol=1e-5)
    assert measured["batch_execution"] == dict(logical_batch=5, requested_sample_chunk_rows=2,
                                              effective_sample_chunk_rows=2, physical_chunks=3)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("implementation", ["python", "native", "libtorch"])
def test_cpu_sample_chunks(case, implementation, dtype, tmp_path):
    compare(case, implementation, "cpu", str(dtype).split(".")[-1], "cpu", tmp_path)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("implementation", ["python", "native", "libtorch"])
def test_mixed_sample_chunks(case, implementation, tmp_path):
    from test_online_consumer_npu import target
    preset = {"pdg": "mixed-a", "timed-dag": "mixed-b", "settle": "mixed-c"}[case[0]]
    compare(case, implementation, target(), "float32", preset, tmp_path)


def test_sample_chunk_refusals_and_whole_batch():
    from test_online_consumer import packet
    p = packet()
    args = dict(family="settle", implementation="python", device="cpu", steps=1, warmup=0)
    for value in (-1, True, 2**63):
        with pytest.raises(ValueError, match="sample-chunk-rows"):
            run(p, sample_chunk_rows=value, **args)
    with pytest.raises(ValueError, match="sample chunking"):
        run(p, sample_chunk_rows=1, preset="resident", **args)
    baseline = run(p, **args)
    larger = run(p, sample_chunk_rows=2**63-1, **args)
    assert baseline["losses"] == larger["losses"]
    assert larger["batch_execution"] == dict(logical_batch=2, requested_sample_chunk_rows=2**63-1,
                                           effective_sample_chunk_rows=2, physical_chunks=1)


@pytest.mark.parametrize("implementation", ["python", "libtorch"])
def test_sample_chunk_shared_launcher(implementation, tmp_path):
    from test_online_consumer import packet
    p = packet("attention")
    source = tmp_path/"packet.json"
    source.write_text(json.dumps(p))
    output = tmp_path/"run"
    cmd = [sys.executable, str(ROOT/"scripts/run_execution_flow.py"), "--packet", str(source),
           "--output-dir", str(output), "--implementation", implementation, "--device", "cpu",
           "--family", "settle", "--preset", "cpu", "--schedule", "prefill", "--training",
           "--optimizer", "adamw", "--steps", "1", "--warmup", "0", "--sample-chunk-rows", "1"]
    if implementation == "libtorch":
        binary = os.environ.get("TIDE_ONLINE_BINARY")
        if not binary:
            pytest.skip("standalone consumer not explicitly selected")
        cmd.extend(("--native-binary", binary))
    completed = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stdout+completed.stderr
    r = json.loads((output/"result.json").read_text())
    assert r["state"] == "passed" and r["batch_execution"]["physical_chunks"] == 2
    assert r["input_tokens_per_step"] == 8 and r["outputs"] == [8]
