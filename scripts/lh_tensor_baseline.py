"""Independent static core-5 LH/HB tensor anchor, pinned to upstream 0770593.

Supports aligned Settle ports, uniform KV configuration inside each region,
qry/sum, Read+, cat, ind/att content_stats, fixed/context cat, total/quad.
All N×B×L Full rows execute before support masking. Output/raw-state numerical
projection only: not complete event observations or None-versus-zero roots.
"""
from dataclasses import dataclass
import torch
from torch.nn import functional as F
from lh_tensor_math import norm, project, describe, select, full
from lh_tensor_state import empty, validate as validate_state, append, decode


@dataclass(frozen=True)
class Sources:
    groups: tuple
    transforms: tuple
    incidence: torch.Tensor


class StaticKV:
    """Compile only static topology and parameter bindings; no trajectory pass.

    Numerical calls retain whole source-region tensors, gather needed source
    rows once per predecessor and reduce physical atoms through incidence.
    Duplicate edges count repeatedly; SrcClass only selects parameters.
    """
    identity = "independent-static-kv-dense-full-v1"

    def __init__(self, binding, spec):
        from tidegraph.weighted import KVConfig
        from tidegraph.next import AdoptNext
        if binding.graph.identity != spec.graph.identity or spec.binding != "ports":
            raise ValueError("LH anchor requires the same aligned Settle ports graph")
        self.binding, self.spec = binding, spec
        self.members = tuple(tuple(v for v, n in enumerate(binding.graph.nodes) if n.region == r)
                             for r in range(len(binding.graph.regions)))
        self.order = sorted(range(len(self.members)), key=lambda r: (spec.ranks[r], r))
        self.location = {v: (r, i) for r, members in enumerate(self.members) for i, v in enumerate(members)}
        if (any(n.clear for n in binding.graph.nodes) or any(not r.observe_all for r in binding.graph.regions)
                or any(type(n.next_program) is not AdoptNext for n in binding.model.nodes)):
            raise ValueError("LH anchor requires observe-all/adopt retained states")
        for r, members in enumerate(self.members):
            configs = [binding.memories[v] for v in members]
            c = binding.regions[r]
            if (not members or not all(isinstance(m, KVConfig) and m == configs[0] for m in configs)
                    or configs[0].memory_read not in {"sum", "qry"}
                    or (c.read_mode, c.local_score, c.selection_mode, c.regularizer, c.support_policy)
                    != ("+", "cat", "regional_total", "quad", "strict")
                    or c.region_control_mode not in {"context", "fixed"}
                    or c.region_control_mode == "context" and c.region_evidence != "cat"
                    or c.region_score not in {"ind", "att"}
                    or c.region_score == "att" and c.region_key != "content_stats"):
                raise ValueError("LH anchor requires homogeneous per-region KV + cat/ind-or-att/total/quad")
        binding.model.validate_binding(binding.graph)
        self.sources = tuple(self._sources(r) for r in range(len(self.members)))
        self.outgoing = torch.tensor([sum(e.source == v for e in binding.graph.edges)
                                      for v in range(len(binding.graph.nodes))], dtype=torch.int64,
                                     device=binding.model.nodes[0].bias.device)

    def _sources(self, r):
        graph, model = self.binding.graph, self.binding.model
        slots, transforms, rows = {}, [], []
        for target_index, target in enumerate(self.members[r]):
            agg = model.nodes[target].aggregate_program
            atoms = [("input", p, self.binding.source_classes.inputs[p])
                     for p, node in enumerate(graph.inputs) if node == target]
            atoms += [("node", e.source, self.binding.source_classes.edges[i])
                      for i, e in enumerate(graph.edges) if e.target == target]
            for kind, source, label in atoms:
                transform = agg.transforms[str(label)]
                # Same input + actual aliased parameters allows shared compute;
                # equal numeric values alone never do.
                key = (kind, source, tuple(id(p) for p in transform.parameters()), agg.alpha, agg.epsilon)
                if key not in slots:
                    slots[key] = len(slots)
                    transforms.append((transform, agg.alpha, agg.epsilon))
                rows.append((target_index, slots[key]))
        if not slots:
            raise ValueError("LH anchor requires each region to have physical sources")
        by_source = {}
        for (kind, source, *_), slot in slots.items():
            owner, index = (-1, source) if kind == "input" else self.location[source]
            by_source.setdefault(owner, []).append((slot, index))
        # Reorder each source owner's slots contiguously to gather whole blocks.
        ordered = [slot for group in by_source.values() for slot, _ in group]
        inverse = {old: new for new, old in enumerate(ordered)}
        incidence = torch.zeros((len(self.members[r]), len(slots)))
        for target, source in rows:
            incidence[target, inverse[source]] += 1
        ref = model.nodes[self.members[r][0]].bias
        groups = tuple((owner, torch.tensor([index for _, index in group], dtype=torch.int64, device=ref.device))
                       for owner, group in by_source.items())
        return Sources(groups, tuple(transforms[slot] for slot in ordered), incidence.to(ref))

    def _aggregate(self, r, x, support, outputs, controls):
        source = self.sources[r]
        values, masses = [], []
        for owner, index in source.groups:
            if owner == -1:
                values.append(x.unsqueeze(0).expand(len(index), -1, -1, -1))
                masses.append(support.index_select(0, index).to(x.dtype))
            else:
                values.append(outputs[owner].index_select(0, index))
                masses.append(controls[owner].index_select(0, index))
        z, mass = torch.cat(values), torch.cat(masses)
        transforms = [row[0] for row in source.transforms]
        epsilons = {row[2] for row in source.transforms}
        alphas = {row[1] for row in source.transforms}
        if len(epsilons) != 1 or len(alphas) != 1:
            raise ValueError("LH anchor region Aggregate alpha/epsilon must agree")
        alpha, epsilon = next(iter(alphas)), next(iter(epsilons))
        if alpha:
            matrices = [t.matrices() for t in transforms]
            if len({tuple(a.shape) for a, _ in matrices}) != 1:
                raise ValueError("LH anchor needs one source hidden width per region")
            hidden = project(norm(z, epsilon), [a for a, _ in matrices])
            hidden = F.silu(hidden+torch.stack([t.embedding for t in transforms])[:, None, None])
            z = z+alpha*project(hidden, [b for _, b in matrices])
        incidence = source.incidence
        total = (incidence@mass.flatten(1)).reshape(len(self.members[r]), *x.shape[:2])
        value = (incidence@(mass[..., None]*z).flatten(1)).reshape(len(self.members[r]), *x.shape)
        return total, value/torch.where(total > 0, total, 1)[..., None]

    def __call__(self, x, *, support=None, start=0, left=None, records=False):
        if (x.ndim != 3 or min(x.shape) < 1 or x.shape[1] > 256
                or x.dtype not in (torch.float32, torch.float64) or not torch.isfinite(x).all()):
            raise ValueError("LH anchor requires finite FP32/FP64 B,L,D with L<=256")
        if type(start) is not int or start < 0 or (start+x.shape[1]-1)*self.spec.stride+max(self.spec.ranks) > 2**63-1:
            raise ValueError("LH anchor start/logical time outside int64 domain")
        model, graph = self.binding.model, self.binding.graph
        if any((n.bias.device, n.bias.dtype, n.kernel.config.width) != (x.device, x.dtype, x.shape[-1]) for n in model.nodes):
            raise ValueError("LH anchor input and parameter shape/dtype/device differ")
        if any((s.incidence.device, s.incidence.dtype) != (x.device, x.dtype) for s in self.sources):
            raise ValueError("rebuild LH static anchor after moving or recasting the bound model")
        if support is None:
            support = torch.ones((len(graph.inputs), *x.shape[:2]), dtype=torch.bool, device=x.device)
        if (support.shape != (len(graph.inputs), *x.shape[:2]) or support.dtype != torch.bool or support.device != x.device):
            raise ValueError("LH anchor physical port support must be bool P,B,L")
        if left is not None and set(left) != set(range(len(self.members))):
            raise ValueError("LH anchor left region domain differs")
        outputs, controls, states, observations = {}, {}, {}, {}
        counts = torch.zeros(3, dtype=torch.int64, device=x.device)
        physical = 0
        for r in self.order:
            nodes = [model.nodes[v] for v in self.members[r]]
            mass, u = self._aggregate(r, x, support, outputs, controls)
            times = (torch.arange(x.shape[1], dtype=torch.int64, device=x.device)+start)*self.spec.stride+self.spec.ranks[r]
            old = empty(nodes, x) if left is None else validate_state(left[r], nodes, x, times[0])
            bank, states[r] = append(nodes, mass, u, times, old)
            fields, visible = describe(nodes, mass, u, bank, times)
            controls[r] = select(model.regions[r], fields)
            outputs[r] = full(nodes, u, bank, times, visible, controls[r])
            active = controls[r].detach() > 0
            degree = self.outgoing[list(self.members[r])][:, None, None]
            counts = counts+torch.stack(((mass > 0).sum(), active.sum(), (active*degree).sum()))
            physical += mass.numel()
            if records:
                observations[r] = dict(controls=controls[r], active=active, candidates=mass > 0, descriptors=fields)
        port_values, port_controls = [], []
        for v in graph.outputs:
            r, i = self.location[v]
            port_values.append(outputs[r][i]); port_controls.append(controls[r][i])
        content, strength = torch.stack(port_values), torch.stack(port_controls)
        hidden = (strength[..., None]*content).sum(0)/strength.sum(0).clamp_min(1)[..., None]
        return dict(hidden=hidden, output_contents=content, output_controls=strength, states=states,
                    regions=observations, counts=counts, physical_full_rows=physical,
                    implementation=self.identity, state_schema="independent-static-kv-bank-v1")

    def decode(self, result):
        return decode(result["states"], self.members)


def run(binding, spec, x, **options):
    """Convenience wrapper; benchmarks should compile StaticKV outside timing."""
    return StaticKV(binding, spec)(x, **options)
