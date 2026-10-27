"""Pinned core-5 examples for target qualification, never a scheduler input trace.

Small FC/HB, feedback PDG and nonaligned TimedDAG include duplicate edges,
zero payload mass, two independent samples and deliberately shared parameters.
Local formula truth anchors remain in the semantic map's independent tests.
"""
from dataclasses import dataclass, replace
import torch
from tidegraph import Edge, External, Graph, Node, Region
from tidegraph.weighted import KVConfig, RecurrentConfig, RegionalConfig, bind, fully_connected, hb_lattice
from tidegraph.weighted.config import ATTENTION, REGIONAL, ALLOCATED


@dataclass(frozen=True)
class Case:
    name: str
    family: str = "kv"
    topology: str = "fc"
    read: str = "+"
    sending: str = "regional_total"
    regularizer: str = "quad"
    attention: bool = False


CASES = [Case("kv-fc"), Case("kv-hb", topology="hb"),
         Case("kv-pdg-initial", topology="pdg", read="0"),
         Case("affine-dag-old", "affine", "dag", "-"),
         Case("delta-pdg", "delta", "pdg"),
         Case("gated-hb", "gated_delta", "hb"), Case("ssd-dag", "ssd", "dag"),
         Case("heterogeneous-attention", "heterogeneous", "hb", attention=True, regularizer="bern"),
         Case("ports-sparse", topology="ports"),
         Case("entropy-shannon", regularizer="ent")]
CASES += [Case("send-"+name, "heterogeneous", "dag", sending=name)
          for name in ("regional_threshold", "regional_total_zero", "regional_threshold_hard",
                       "hard_selection", "constrained", "proportional", "unit_attention",
                       "direct_attention", "pooled_attention")]


def fixture(case, dtype=torch.float32):
    if case.topology in {"fc", "hb", "ports"}:
        spec = hb_lattice(2, 1, 2, offsets=((0, 1), (0,))) if case.topology == "hb" else fully_connected(2, 2)
        graph, times, stop = spec.graph, (1, spec.stride+1), 2*spec.stride
        if case.topology == "ports":
            stop = 3*spec.stride
    else:
        edges = (Edge(0, 1, 1), Edge(0, 1, 1), Edge(0, 1, 3), Edge(1, 2, 2))
        if case.topology == "pdg":
            edges += (Edge(2, 0, 1),)
        graph = Graph((Node(0), Node(0), Node(1)), edges, (Region(2), Region(1)), (0, 1), (1, 2))
        times, stop = (0, 3), 8
    options = [KVConfig(width=2, summary_width=2, ffn_width=3, window=7, epsilon=.3,
                        full_frequencies=((.2,),))]
    options += [RecurrentConfig(memory_family=f, width=2, summary_width=2, key_width=2,
                  value_width=2, ffn_width=3, epsilon=.3, memory_read="state",
                  **({} if f in {"affine", "delta"} else dict(conv_size=3, conv_stride=2)))
                for f in ("affine", "delta", "gated_delta", "ssd")]
    names = ("kv", "affine", "delta", "gated_delta", "ssd")
    memories = tuple(options[i % 5] if case.family == "heterogeneous" else options[names.index(case.family)]
                     for i in range(len(graph.nodes)))
    regions = []
    for layout in graph.region_layouts:
        fields = dict(selection_mode=case.sending, read_mode=case.read, gamma_max=1.6, hidden_width=3)
        if case.sending not in ATTENTION:
            fields["k"] = len(layout.members)
        if case.sending not in REGIONAL:
            fields["send_gain"] = "context"
        if case.sending in ALLOCATED:
            fields["regularizer"] = case.regularizer
            if case.regularizer == "ent":
                fields["ent_alpha"] = 1.
        if case.attention:
            fields.update(local_score="match", region_score="att", region_key="content_stats",
                          region_evidence="grouped", query_weight="received", selection_width=2)
        regions.append(RegionalConfig(**fields))
    binding = bind(graph, memories=memories, regions=tuple(regions), alpha_agg=.2, dtype=dtype)
    if case.attention:
        for node in binding.model.nodes:
            read = node.read_program
            read.key = read.score.encoders["mem"]
            read.evidence.encoders["mem"] = read.score.encoders["mem"]
    xs = []
    for sample in range(2):
        for port in range(len(graph.inputs)):
            for pos, time in enumerate(times):
                if case.topology == "ports" and port == 1 and pos == 1:
                    time += spec.stride  # Different legal global input positions per port.
                mass = 0. if sample == 1 and port == 0 and pos == 0 else .35+.1*port
                value = torch.tensor([mass, .3+.2*pos, -.4+.1*sample], dtype=dtype, requires_grad=True)
                xs.append(External(sample, port, pos, time, value))
    return binding, xs, stop


def inputs_to(xs, device):
    leaves = {id(x.value): x.value.detach().to(device).clone().requires_grad_() for x in xs}
    return [replace(x, value=leaves[id(x.value)]) for x in xs]


def leaves(binding, xs):
    return {n: p for n, p in binding.model.named_parameters() if p.requires_grad} | {
        f"input.{i}": x.value for i, x in enumerate(xs)}
