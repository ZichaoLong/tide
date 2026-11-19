"""Intrusive, untimed semantic work/parameter audit of the active evaluator.

An atom is a unique Parameter or one disjoint owner/group/source row of it.
Effective touches include every positive reception's preparation and regional
work and every positive Full, independent of whether a numerical gradient is
zero. Padding and physical matrix rows are separate observations. No parameter
is called active merely because a padded batched operation loaded its tensor.

Settle deduplicates within (sample, input step), across all settled layers.
Timed deduplicates within (sample, logical tick), across all regions; cumulative
per-input work and cross-time union are also reported, without causal attribution.
These hooks copy supports to CPU and are deliberately excluded from timing.
"""
from collections import defaultdict
import importlib

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from tide.parameters import HeadQuery, NodeLinear, NodeVector, owner_ids
from tide.reception import SourceEmbedding


class RuntimeAudit(TorchDispatchMode):
    """A separate ordinary forward's operators/dtypes and materialized bytes.

    Run without WorkAudit hooks: their CPU support copies are diagnostics and
    must not be counted as model movement. Output byte totals describe logical
    materializations, not measured hardware traffic or complete memory reads.
    """
    def __init__(self):
        self.operators = defaultdict(int)
        self.matrix_shapes = defaultdict(int)
        self.movement = defaultdict(lambda: dict(calls=0, output_bytes=0))

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        name = str(func)
        tensors = [x for x in tree_flatten(result)[0] if isinstance(x, torch.Tensor)]
        if any(token in name for token in ("mm.", "bmm.", "addmm.", "matmul.", "linear.", "attention", "convolution")):
            inputs = [x for x in tree_flatten((args, kwargs))[0] if isinstance(x, torch.Tensor)]
            before = tuple(sorted({str(x.device)+":"+str(x.dtype) for x in inputs}))
            after = tuple(sorted({str(x.device)+":"+str(x.dtype) for x in tensors}))
            self.operators[name, before, after] += 1
            if any(name.startswith("aten."+op+".") for op in ("mm", "bmm", "addmm")):
                self.matrix_shapes[name, tuple(tuple(x.shape) for x in inputs)] += 1
        if any(name.startswith("aten."+token+".") for token in
               ("_to_copy", "copy_", "clone", "contiguous", "cat", "index", "index_put", "index_put_",
                "index_select", "gather", "scatter", "scatter_", "index_copy", "index_copy_")):
            row = self.movement[name]
            row["calls"] += 1
            row["output_bytes"] += sum(x.numel()*x.element_size() for x in tensors)
        return result

    def report(self):
        return dict(operators=[dict(operator=op, inputs=list(before), outputs=list(after), calls=count)
                               for (op, before, after), count in sorted(self.operators.items())],
                    materializations=dict(self.movement),
                    matrix_shapes=[dict(operator=op, inputs=[list(shape) for shape in shapes], calls=count)
                                   for (op, shapes), count in sorted(self.matrix_shapes.items())],
                    materialization_definition="logical output bytes from dispatched copy/index/concatenate operations; not hardware bus traffic, bandwidth or all operand loads",
                    instrumentation="separate ordinary forward without semantic counting hooks; excluded from timing",
                    bf16_matrix_observed=any("bfloat16" in dtype for (_, _, after) in self.operators for dtype in after))


class ParameterInventory:
    def __init__(self, model):
        self.model = model
        self.parameters = dict(model.named_parameters())
        self.names = {id(p): name for name, p in self.parameters.items()}
        self.sizes, self.classes = {}, {}
        self.owner_atoms, self.source_atoms, self.whole_atoms = {}, {}, {}
        graph = model.graph
        self.local_nodes = {"receive": list(range(len(graph.nodes))),
                            "describe": [v for v, _ in sorted(model.description_ids.items(), key=lambda x: x[1])]}
        self.local_nodes.update({"memories."+kind: [v for v, _ in sorted(ids.items(), key=lambda x: x[1])]
                                 for kind, ids in model.memory_ids.items()})
        handled = set()
        for path, module in model.named_modules():
            prefix = next((p for p in self.local_nodes if path.startswith(p+".")), None)
            if isinstance(module, (NodeLinear, NodeVector)):
                owners = self.local_nodes[prefix]
                groups = module.groups.detach().cpu().tolist()
                rows = [set() for _ in owners]
                for leaf, parameter in module.named_parameters(recurse=False):
                    handled.add(id(parameter))
                    name = self.names[id(parameter)]
                    private = (leaf in {"left", "right"} or path == "describe.offset" or
                               isinstance(module, NodeLinear) and not module.shared or
                               isinstance(module, NodeVector) and not module.shared)
                    indices = list(range(len(owners))) if leaf in {"left", "right"} else groups
                    for owner, index in enumerate(indices):
                        classification = graph.nodes[owners[owner]].role if private else "base"
                        atom = (name, index)
                        self._atom(atom, parameter[0].numel(), classification)
                        rows[owner].add(atom)
                self.owner_atoms[path] = rows
            elif isinstance(module, SourceEmbedding):
                parameter = module.weight
                handled.add(id(parameter))
                name = self.names[id(parameter)]
                lookup = module.lookup.detach().cpu().tolist()
                mapping = {}
                for owner, classes in enumerate(lookup):
                    for label, index in enumerate(classes):
                        if index >= 0:
                            atom = (name, index)
                            classification = (graph.nodes[self.local_nodes[prefix][owner]].role
                                              if model.config.sharing == "independent" else "base")
                            self._atom(atom, parameter.shape[1], classification)
                            mapping[owner, label] = atom
                self.source_atoms[path] = mapping
        for name, parameter in self.parameters.items():
            if id(parameter) not in handled:
                atom = (name, None)
                self._atom(atom, parameter.numel(), "ingress" if name.startswith("inputs.") else "base")
                self.whole_atoms[name] = atom
        if sum(self.sizes.values()) != sum(p.numel() for p in self.parameters.values()):
            raise AssertionError("parameter atom inventory is not an exact disjoint partition")

    def _atom(self, atom, size, classification):
        if atom in self.sizes and (self.sizes[atom] != size or self.classes[atom] != classification):
            raise AssertionError("inconsistent shared parameter classification")
        self.sizes[atom], self.classes[atom] = size, classification

    def nodes(self, prefixes, owner):
        return set().union(*(rows[owner] for name, rows in self.owner_atoms.items()
                             if any(name == prefix or name.startswith(prefix+".") for prefix in prefixes)))

    def whole(self, prefix):
        return {atom for name, atom in self.whole_atoms.items()
                if name == prefix or name.startswith(prefix+".")}

    def count(self, atoms):
        return sum(self.sizes[atom] for atom in atoms)

    def report(self):
        counts = {group: sum(size for atom, size in self.sizes.items() if self.classes[atom] == group)
                  for group in ("base", "D", "H", "ingress")}
        fraction = ((self.model.config.budget or max(1, self.model.graph.config.n//32)) / self.model.graph.config.n
                    if self.model.config.allocation == "quad" else 1.0)
        return dict(unique_parameters=sum(self.sizes.values()),
                    scope="all unique trainable backbone parameters including enabled ingress projections; no task head",
                    ingress_parameters=counts["ingress"],
                    textbook=dict(base=counts["base"], private_D=counts["D"], private_H=counts["H"],
                                  total_excluding_ingress=counts["base"]+counts["D"]+counts["H"],
                                  active_proxy=counts["base"]+fraction*counts["D"]+counts["H"],
                                  D_fraction=fraction, is_actual_touched=False,
                                  assumption="all regions receive; all H act; D private scaled by k/n only for quad"))


class WorkAudit:
    """Observe one model in an exclusive Python context, outside any timing."""
    def __init__(self, model, *, inventory=None):
        self.model = model
        self.inventory = ParameterInventory(model) if inventory is None else inventory
        self.events = defaultdict(lambda: defaultdict(set))
        self.domain = set()
        self.work = defaultdict(int)
        self.work_events = defaultdict(lambda: defaultdict(int))
        self.matrices = defaultdict(lambda: dict(calls=0, rows=0, dense_multiply_adds=0))
        self.matrix_dispatch = defaultdict(lambda: dict(calls=0, base_weight_rows=0,
                                                       unique_base_weight_rows=0, repeated_base_weight_rows=0,
                                                       bank_view_span_rows=0))
        self.packing = defaultdict(lambda: dict(calls=0, selected_rows=0, executed_rows=0))
        self.batch_inputs = 0
        self._originals = []
        self._handles = []

    def _wrap(self, obj, name, callback):
        original, own = getattr(obj, name), name in vars(obj)
        def run(*args, **kwargs):
            return callback(original, *args, **kwargs)
        self._originals.append((obj, name, original, own))
        setattr(obj, name, run)

    def _add(self, stage, key, atoms):
        self.events[stage][key].update(atoms)

    def _work(self, name, values, offset=0):
        values = values.detach().cpu()
        self.work[name] += int(values.sum())
        for sample, row in enumerate(values.tolist()):
            for tick, value in enumerate(row):
                self.work_events[sample, self.tick_start+offset+tick][name] += int(value)

    def _node_events(self, stage, mask, owners, prefixes):
        support = mask.detach().cpu()
        ids = owner_ids(owners).detach().cpu().tolist()
        atoms = [self.inventory.nodes(prefixes, i) for i in ids]
        for node, sample, tick in support.nonzero().tolist():
            self._add(stage, (sample, self.tick_start+tick), atoms[node])

    def _run(self, original, x, state=None, *, record_work=False, inject=True):
        batch, inputs = x.shape[:2]
        self.tick_start = (0 if state is None else state.position) * (1 if self.model.settle else self.model.graph.period)
        self.length = inputs * (1 if self.model.settle else self.model.graph.period)
        self.region_index = 0
        self.domain.update((b, self.tick_start+t) for b in range(batch) for t in range(self.length))
        if inject:
            self.batch_inputs += batch*inputs
            for b in range(batch):
                for t in range(inputs):
                    key = b, self.tick_start+t*(1 if self.model.settle else self.model.graph.period)
                    self._add("ingress", key, self.inventory.whole("inputs"))
        return original(x, state, record_work=record_work, inject=inject)

    def _receive(self, original, content, mass, classes, owners, edge_mask=None):
        self.region = self.region_index
        self.region_index += 1
        support = mass.detach().cpu() > 0
        if edge_mask is not None:
            support &= edge_mask.detach().cpu()[:, None, None]
        candidate = support.any(-1)
        ids = owner_ids(owners).detach().cpu().tolist()
        labels = classes.detach().cpu()
        self._node_events("reception", candidate, owners, ["receive"])
        for node, sample, tick, edge in support.nonzero().tolist():
            atoms = {mapping[ids[node], int(labels[node, edge])]
                     for mapping in self.inventory.source_atoms.values()}
            self._add("reception", (sample, self.tick_start+tick), atoms)
        self._work("positive_messages", support.sum((0, 3)))
        self._work("message_slots", torch.full(candidate.shape[1:], support.shape[0]*support.shape[-1]))
        self._work("candidate_updates", candidate.sum(0))
        self._work("candidate_slots", torch.full(candidate.shape[1:], candidate.shape[0]))
        if self.model.config.aggregation != "source_residual":
            self._work("reception_attention_positive_pairs", support.sum(-1).square().sum(0)*self.model.receive.heads)
        result = original(content, mass, classes, owners, edge_mask)
        if self.model.receive.effective_qkv_backend == "fused":
            # Fused projections bypass child NodeLinear.forward hooks. Count
            # the unchanged mathematical matrices/adapters and padded rows.
            for name in ("q", "k", "v"):
                self._matrix("receive."+name, getattr(self.model.receive, name), (content, owners), result,
                             record_dispatch=False)
            self._base_matrix_dispatch("receive.fused_qkv", self.model.receive.q, (content, owners),
                                       out_features=3*self.model.receive.q.out_features)
        return result

    def _describe(self, original, u, received, memories, owners):
        self._node_events("regional", received > 0, owners, ["describe"])
        return original(u, received, memories, owners)

    def _regional(self, number, original, z, score, weight):
        self._regional_work(number, weight)
        return original(z, score, weight)

    def _regional_work(self, number, weight):
        support = weight.detach().cpu() > 0
        atoms = self.inventory.whole("regional."+str(number))
        for sample, tick in support.any(0).nonzero().tolist():
            self._add("regional", (sample, self.tick_start+tick), atoms)
        self._work("regional_attention_positive_pairs", support.sum(0).square()*self.model.regional[number].heads)

    def _regional_batch(self, original, regions, z, score, weight):
        for i, region in enumerate(regions):
            self._regional_work(self.model.region_group[region], weight[i])
        return original(regions, z, score, weight)

    def _prepare(self, kind, original, u, mass, times, owners, state=None):
        names = (["kv", "summary"] if kind == "K" else
                 ["input", "summary", "read", "a_log", "dt_bias", "conv_kernel", "conv_bias"])
        self._node_events("memory_prepare", mass > 0, owners, ["memories."+kind+"."+name for name in names])
        return original(u, mass, times, owners, state)

    def _finish(self, kind, original, context, control):
        names = ["query", "output", "ffn"] if kind == "K" else ["skip", "rho", "output", "ffn"]
        self._node_events("full", control > 0, context[2], ["memories."+kind+"."+name for name in names])
        self._work("full_actions", (control.detach() > 0).sum(0))
        if kind == "K":
            times, bank = context[1].detach().cpu(), context[3]
            key_times, key_weight = bank.times.detach().cpu(), bank.weight.detach().cpu()
            active = control.detach().cpu() > 0
            window = self.model.memories[kind].window
            for start in range(0, times.shape[-1], 64):
                delta = times[..., start:start+64, None]-key_times[..., None, :]
                mask = (delta >= 0) & (delta < window) & (key_weight[..., None, :] > 0) & active[..., start:start+64, None]
                self._work("kv_attention_effective_pairs", mask.sum((0, 3))*self.model.memories[kind].heads, offset=start)
        return original(context, control)

    def _packed(self, kind, original, *args, **kwargs):
        result = original(*args, **kwargs)
        record = self.packing[kind]
        record["calls"] += 1
        if result is not None:
            record["selected_rows"] += result.node.numel()
            record["executed_rows"] += result.data.numel() // result.data.shape[-1]
        return result

    def _matrix(self, path, module, inputs, output, *, record_dispatch=True):
        x = inputs[0]
        rows = x.numel() // module.in_features
        record = self.matrices[path]
        record["calls"] += 1
        record["rows"] += rows
        record["dense_multiply_adds"] += rows * module.in_features * module.out_features
        if module.shared:
            rank = module.left.shape[-1]
            record["dense_multiply_adds"] += rows * rank * (module.in_features+module.out_features)
        if record_dispatch:
            self._base_matrix_dispatch(path, module, inputs)

    def _base_matrix_dispatch(self, path, module, inputs, *, out_features=None):
        x = inputs[0]
        rows = x.numel()//module.in_features
        owners = inputs[1] if len(inputs) > 1 else None
        uniform = module._uniform_group(owners)
        n = x.shape[0]
        if uniform is None:
            ids = list(range(module.nodes)) if owners is None else owner_ids(owners).detach().cpu().tolist()
            unique = len({module._host_groups[index] for index in ids})
            weight_rows, operator = n, "bmm"
            bank_span = 0
            if module.weight_access == "grouped":
                possible = getattr(owners, "possible_ids", tuple(range(module.nodes)))
                groups = [module._host_groups[index] for index in possible]
                bank_span = max(groups)-min(groups)+1 if groups else module.weight.shape[0]
                weight_rows = unique
                operator = "npu_grouped_matmul" if x.device.type == "npu" else "cpu_segmented_mm"
        else:
            weight_rows, unique, operator = 1, 1, "mm"
            bank_span = 0
        width = module.out_features if out_features is None else out_features
        key = (path, operator, n, rows//n, module.in_features, width)
        dispatch = self.matrix_dispatch[key]
        dispatch["calls"] += 1
        dispatch["base_weight_rows"] += weight_rows
        dispatch["unique_base_weight_rows"] += unique
        dispatch["repeated_base_weight_rows"] += weight_rows-unique
        dispatch["bank_view_span_rows"] += bank_span

    def _matrix_dispatch_report(self):
        return [dict(path=path, base_operator=operator, node_rows=nodes, rows_per_node=rows,
                     in_features=inputs, out_features=outputs, **counts)
                for (path, operator, nodes, rows, inputs, outputs), counts in sorted(self.matrix_dispatch.items())]

    def _ffn_packing_report(self):
        # Query packing still describes the unchanged Attention layout. The
        # actual FFN gate operands may instead use compacted owner buckets.
        result = {}
        for kind, packed in self.packing.items():
            prefix = "memories."+kind+".ffn.gate"
            matrices = [row for name, row in self.matrices.items()
                        if name == prefix or name.startswith(prefix+".shards.")]
            result[kind] = dict(selected_rows=packed["selected_rows"],
                                executed_rows=sum(row["rows"] for row in matrices),
                                calls=sum(row["calls"] for row in matrices),
                                query_rectangular_rows=packed["executed_rows"],
                                policy=self.model.config.ffn_packing)
        return result

    def _head_matrix(self, path, module, inputs, output):
        if not module.uses_fused_projection(inputs[0]):
            return
        # Child NodeLinear.forward hooks do not run in the fused contraction.
        # Count its complete mathematical matrices/adapters once, including pad.
        for index, head in enumerate(module.heads):
            self._matrix(path+f".heads.{index}",head,inputs,output,record_dispatch=False)
        self._base_matrix_dispatch(path+".fused_heads", module.heads[0], inputs,
                                   out_features=sum(head.out_features for head in module.heads))

    def __enter__(self):
        from functools import partial
        self._wrap(self.model, "_run", self._run)
        self._wrap(self.model.receive, "forward", self._receive)
        self._wrap(self.model.describe, "forward", self._describe)
        self._wrap(self.model, "_score_regions", self._regional_batch)
        for i, region in enumerate(self.model.regional):
            self._wrap(region, "forward", partial(self._regional, i))
        for kind, memory in self.model.memories.items():
            self._wrap(memory, "prepare", partial(self._prepare, kind))
            self._wrap(memory, "finish", partial(self._finish, kind))
            module = importlib.import_module("tide.kv" if kind == "K" else "tide.ssd_memory")
            self._wrap(module, "select_queries", partial(self._packed, kind))
        for path, module in self.model.named_modules():
            if isinstance(module, NodeLinear):
                self._handles.append(module.register_forward_hook(partial(self._matrix, path)))
            elif isinstance(module, HeadQuery):
                self._handles.append(module.register_forward_hook(partial(self._head_matrix,path)))
        return self

    def __exit__(self, *exc):
        for handle in self._handles:
            handle.remove()
        for obj, name, original, owned in reversed(self._originals):
            if owned:
                setattr(obj, name, original)
            else:
                delattr(obj, name)

    def report(self):
        combined = defaultdict(set)
        stages = {}
        for stage, events in self.events.items():
            for key, atoms in events.items():
                combined[key].update(atoms)
            stages[stage] = self._summary(events)
        union = set().union(*combined.values()) if combined else set()
        total = self._summary(combined)
        total["unique_across_time_and_batch"] = self.inventory.count(union)
        total["cumulative_per_input_position"] = total["sum"] / self.batch_inputs if self.batch_inputs else None
        return dict(parameters=self.inventory.report(), effective_touched=total, stages=stages,
                    definition="unique Parameter scalars in semantically effective owner/group/source slices; not nonzero gradients or physical loads",
                    deduplication=("within each sample and input step across settled layers" if self.model.settle else
                                   "within each sample and logical tick across regions; repeated ticks accumulate; no output causal attribution"),
                    input_positions=self.batch_inputs, sample_time_slots=len(self.domain),
                    logical_axis="input_step" if self.model.settle else "tick",
                    work=dict(self.work),
                    work_per_input_position={name: count/self.batch_inputs for name, count in self.work.items()} if self.batch_inputs else {},
                    work_by_sample_time=[dict(sample=b, time=t, **self.work_events[b, t]) for b, t in sorted(self.domain)],
                    query_packing=dict(self.packing),
                    ffn_packing=self._ffn_packing_report(),
                    node_matrices=dict(self.matrices),
                    node_matrix_dispatch=self._matrix_dispatch_report(),
                    matrix_dispatch_definition="NodeLinear base dispatch; gather rows describe BMM operands, grouped rows count nonempty unique groups with separate bank-view span including empty groups; neither is copied bytes or bus traffic; private adapters remain separate in matrix cost",
                    matrix_cost_definition="multiply-add pairs for actually executed NodeLinear rows, including padding/adapters; excludes attention/normalization/non-NodeLinear ops; not complete FLOPs")

    def _summary(self, events):
        counts = [self.inventory.count(events.get(key, ())) for key in sorted(self.domain)]
        return dict(sum=sum(counts), mean=sum(counts)/len(counts) if counts else 0,
                    min=min(counts, default=0), max=max(counts, default=0),
                    by_sample_time=[dict(sample=b, time=t, parameters=count)
                                    for (b, t), count in zip(sorted(self.domain), counts)])


def audit_model(model, x, *, state=None, flush=False):
    """Return result, optional flush result and an untimed audit in current mode."""
    with WorkAudit(model) as audit:
        result = model(x, state, record_work=True)
        tail = model.flush(result.state, record_work=True) if flush and not model.settle else None
    return result, tail, audit.report()
