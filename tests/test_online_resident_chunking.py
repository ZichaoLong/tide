"""Previously refused D32 model: automatic physical splits preserve whole training."""
import json
import os
import subprocess

import pytest
import torch
from test_online_consumer_npu import target
from online_consumer_support import same, observer
from flow_topology import ranked_graph
from flow_protocol import make_continuous_packet, native_text
from tools.online_bench.host import run
from tidegraph import ResidentLimits, ResidentPlacement, ResidentTrainingLimits


@pytest.mark.parametrize("implementation", ["native", "libtorch"])
def test_d32_automatic_reverse_split(implementation, tmp_path):
    device = target()
    p = make_continuous_packet(graph=ranked_graph(layers=3, region_width=4, fanout=2, local_span=2),
        memory="attention", width=32, batch=2, tokens=3, vocab=17, clear=False)
    expected = []
    common = dict(family="timed-dag", training=True, optimizer="adamw", steps=2, warmup=0,
                  windows_per_step=2, diagnostics=True)
    reference = run(p, implementation="python", device="cpu", schedule="streaming",
                    observer=observer(expected), **common)
    results = []
    for maximum in (16, 1):
        actual = []
        if implementation == "native":
            candidate = run(p, implementation="native", device=device, schedule="prefill", preset="resident",
                resident_placement=ResidentPlacement(devices=(str(device), f"npu:{device.index+1}")),
                resident_limits=ResidentLimits(queue=128, arrivals=128, outputs=128, trace=512,
                    kv_trace_rows=1024, workspace_bytes=512*1024**2, chunk_policy="aggressive"),
                training_limits=ResidentTrainingLimits(windows=2, backward_bytes=2*1024**3,
                    reverse_chunk_rows=maximum), native_library=os.environ["TIDE_BUILD_DIR"],
                resident_library=os.environ["TIDE_RESIDENT_LIBRARY"], observer=observer(actual), **common)
        else:
            out = tmp_path / f"rows{maximum}"
            packet = tmp_path / f"packet{maximum}.txt"; packet.write_text(native_text(p))
            command = [os.environ["TIDE_ONLINE_BINARY"], "--device="+str(device), "--dtype=float32",
                "--packet="+str(packet), "--output-dir="+str(out), "--family=timed-dag", "--preset=resident",
                "--schedule=prefill", "--training", "--optimizer=adamw", "--steps=2", "--warmup=0",
                "--windows-per-step=2", "--diagnostics", "--devices=2", "--chunk-policy=aggressive",
                "--resident-trace=512", "--resident-kv-trace-rows=1024", "--resident-reverse-chunk-rows="+str(maximum)]
            done = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
            assert done.returncode == 0, done.stdout
            candidate = json.loads((out/"result.json").read_text())
            actual = [json.loads(line) for line in (out/"diagnostics.jsonl").read_text().splitlines()]
        same(actual, expected)
        assert candidate["outputs"] == reference["outputs"] and candidate["final_cut"] == reference["final_cut"]
        torch.testing.assert_close(torch.tensor(candidate["losses"]), torch.tensor(reference["losses"]), atol=1e-6, rtol=1e-5)
        stats = candidate["statistics"]
        for s in stats:
            assert s["reverse_fiber_requested_max"] == maximum
            assert 1 <= s["reverse_fiber_query_rows_min"] <= s["reverse_fiber_query_rows_max"] <= maximum
            assert s["reverse_fiber_tensor_bytes"] <= s["reverse_fiber_budget_bytes"]
            if maximum == 16:
                assert s["reverse_fiber_query_rows_min"] < maximum
        results.append(candidate)
    (tmp_path/"splits.json").write_text(json.dumps(dict(packet=p, reference=reference, candidates=results)))
