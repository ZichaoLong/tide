import json
from pathlib import Path
import subprocess
import torch
from tidegraph import Continuation, Edge, External, Graph, Node, Region
from tidegraph.ops import Model
from tidegraph.reference import run


def test_standalone_libtorch_forward_backward(dtype, tmp_path):
    import _tide_native
    binary = Path(_tide_native.__file__).with_name("tidegraph-smoke")
    name = str(dtype).split(".")[-1]
    out = tmp_path / "smoke"
    result = subprocess.run([str(binary), "--device=cpu", f"--dtype={name}", "--output-dir", str(out)],
                            text=True, capture_output=True, check=True)
    actual = json.loads(result.stdout)
    g = Graph((Node(0),), (Edge(0, 0, 2),), (Region(1),), (0,), (0,))
    m = Model(g, width=2, dtype=dtype)
    with torch.no_grad():
        m.nodes[0].decay.zero_(); m.nodes[0].bias.zero_(); m.nodes[0].read.fill_(1)
        m.nodes[0].weight.copy_(torch.eye(2, dtype=dtype) * 0.2)
        for group in (m.input_scale, m.agg_scale, m.edge_scale, m.output_scale):
            for p in group:
                p.fill_(1)
    x = torch.ones(2, dtype=dtype, requires_grad=True)
    expected = run(g, m, Continuation(g.identity, 1), [External(0, 0, 0, 0, x)], 5, sealed_until=5)
    loss = expected.continuation.states[0, 0].value.sum() + expected.continuation.pending[0].value.sum()
    loss = loss + sum(v.sum() for _, _, _, v in expected.outputs)
    gradient, = torch.autograd.grad(loss, x)
    torch.testing.assert_close(torch.tensor(actual["loss"], dtype=dtype), loss)
    torch.testing.assert_close(torch.tensor(actual["gradient_sum"], dtype=dtype), gradient.sum())
    assert json.loads((out / "result.json").read_text()) == actual
    for options in (["--device=npu"], ["--device=cuda"], ["--device=cpu", "--dtype=float16"],
                    ["--seed=-1"], ["--output-dir", str(out)], ["--unknown"]):
        assert subprocess.run([str(binary), *options], capture_output=True).returncode != 0


def test_cpp_custom_state_kernel(dtype):
    import _tide_native
    binary = Path(_tide_native.__file__).with_name("tidegraph-kernel-check")
    result = subprocess.run([str(binary), "--device=cpu", "--dtype=" + str(dtype).split(".")[-1]],
                            text=True, capture_output=True, check=True)
    assert result.stdout.strip() == "custom-state-kernel: passed"


def test_cpp_custom_full_kernel(dtype):
    import _tide_native
    binary = Path(_tide_native.__file__).with_name("tidegraph-full-check")
    result = subprocess.run([str(binary), "--device=cpu", "--dtype=" + str(dtype).split(".")[-1]],
                            text=True, capture_output=True, check=True)
    assert result.stdout.strip() == "custom-full-kernel: passed"
