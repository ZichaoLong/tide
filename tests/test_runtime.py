"""Backend-neutral runtime and model-placement contracts."""
import torch

from tidegraph import Edge, Graph, Node, Region
from tidegraph.ops import Model
from tidegraph.runtime import resolve_device


def test_cpu_resolution_and_index_normalization():
    device, reason = resolve_device("cpu")
    assert device == torch.device("cpu")
    assert reason == "explicit:cpu"
    device, reason = resolve_device("cpu:0")
    assert device == torch.device("cpu")
    assert reason == "explicit:cpu"


def test_model_device_boundary_preserves_cpu_contract():
    graph = Graph((Node(0),), (Edge(0, 0, 1),), (Region(1),), (0,), (0,))
    model = Model(graph, width=3, dtype=torch.float32, device="cpu")
    assert {parameter.device.type for parameter in model.parameters()} == {"cpu"}
    assert model.nodes[0].bias.dtype == torch.float32


def test_runtime_rejects_negative_accelerator_index():
    try:
        resolve_device("npu", -1)
    except (RuntimeError, ValueError) as error:
        assert "nonnegative" in str(error) or "torch_npu" in str(error) or "NPU" in str(error)
    else:
        raise AssertionError("negative NPU index was accepted")
