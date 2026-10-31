"""Finite core-5 workloads, independent of any candidate's execution trace.

The family, graph, scale and seed identify a workload. All candidates receive
the same physical input records and parameters. Logical windows do not reset
memory; training explicitly detaches only at the optimizer boundary.
"""
from dataclasses import asdict, dataclass, replace
from collections import Counter
import hashlib
import json
import torch
from tidegraph import Edge, External, Graph, Node, Region
from tidegraph.weighted import KVConfig, RecurrentConfig, RegionalConfig, bind, hb_lattice


@dataclass(frozen=True)
class Workload:
    topology: str = "hb"
    family: str = "kv"
    width: int = 4
    samples: int = 1
    positions: int = 2
    seed: int = 7

    def __post_init__(self):
        if self.topology not in {"hb", "dag", "pdg"} or self.family not in {"kv", "ssd"}:
            raise ValueError("unknown weighted benchmark graph/memory")
        if not (2 <= self.width <= 128 and self.width % 2 == 0
                and 1 <= self.samples <= 16 and 1 <= self.positions <= 16
                and 0 <= self.seed < 2**31):
            raise ValueError("weighted benchmark exceeds the finite dimension domain")


def fixture(config):
    if config.topology == "hb":
        spec = hb_lattice(2, 1, 2, offsets=((0, 1), (0,)))
        graph, stride, origin = spec.graph, spec.stride, 1
    else:
        edges = (Edge(0, 1, 1), Edge(0, 1, 1), Edge(0, 1, 3), Edge(1, 2, 2))
        if config.topology == "pdg":
            edges += (Edge(2, 0, 1),)
        graph = Graph((Node(0), Node(0), Node(1)), edges, (Region(2), Region(1)), (0, 1), (1, 2))
        stride, origin = 5, 0
    span = config.positions * stride
    width, summary = config.width, config.width // 2
    if config.family == "kv":
        memory = KVConfig(width=width, summary_width=summary, adapter_rank=2,
                          ffn_width=2*width, window=2*span, epsilon=.3)
    else:
        memory = RecurrentConfig(memory_family="ssd", memory_read="state", width=width,
            summary_width=summary, key_width=width//2, value_width=width//2,
            heads=2, groups=1, ffn_width=2*width, conv_size=3,
            conv_stride=stride if config.topology == "hb" else 2, epsilon=.3)
    regions = tuple(RegionalConfig(k=len(layout.members), hidden_width=max(4, width//2),
                                  gamma_max=1.6) for layout in graph.region_layouts)
    binding = bind(graph, memories=memory, regions=regions, alpha_agg=.2,
                   source_hidden_width=width, dtype=torch.float32, seed=config.seed)
    inputs = []
    for sample in range(config.samples):
        for position in range(config.positions):
            # One real broadcast tensor per sample/position, not equal-value
            # deduplication. The source parameter class does not merge atoms.
            value = torch.cat((torch.tensor([.7]), .2*torch.sin(
                torch.arange(width, dtype=torch.float32)*.19 + .3*position + .1*sample)))
            for port in range(len(graph.inputs)):
                inputs.append(External(sample, port, position, origin+stride*position, value))
    return binding, inputs, span


def window(inputs, span, positions, step, device):
    # Positions are contiguous occurrence counters per physical sample/port;
    # sparse global logical positions determine times, not these counters.
    occurrences = Counter((x.batch, x.port) for x in inputs)
    copies = {}
    for x in inputs:
        if id(x.value) not in copies:
            copies[id(x.value)] = x.value.detach().to(x.value.device if device is None else device).clone().requires_grad_()
    return [replace(x, time=x.time+step*span, position=x.position+step*occurrences[x.batch, x.port],
                    value=copies[id(x.value)]) for x in inputs]


def fingerprint(binding, inputs, config):
    digest = hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode())
    digest.update(binding.graph.identity.encode())
    for name, value in sorted(binding.model.state_dict().items()):
        digest.update(name.encode()); digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().contiguous().cpu().numpy().tobytes())
    for x in inputs:
        digest.update(json.dumps([x.batch, x.port, x.position, x.time]).encode())
        digest.update(x.value.numpy().tobytes())
    return digest.hexdigest()


def limits(binding, config, span):
    # Static conservative admission; no CPU execution supplies a real trace.
    return dict(queue=max(64, 3*len(binding.graph.inputs)*config.samples*config.positions),
                stages=max(32, span*len(binding.graph.regions)), cache=2*span+2,
                tensor_bytes=1024**3, operator_bytes=128*1024**2, timeout_ms=60000,
                retained_bytes=1024**3, backward_bytes=2*1024**3,
                optimizer_bytes=256*1024**2, total_bytes=5*1024**3)
