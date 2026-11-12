"""LH task projection, actual column decoding and strictly classified checks.

This diagnostic intentionally materializes B×nodes raw slices for observation
and its task VJP. It is not the benchmark path or an all-intermediate-root test.
No reference computation participates in decoding a candidate.
"""
import torch
from benchmark_lh_tensor import ports, output_values
from lh_tensor_state import FLOATS


def raw_columns(columns):
    """Read only actual terminal banks, clocks and inert fields from columns."""
    fields = {}
    for v, state in sorted(columns.states.items()):
        for b, (a, z) in enumerate(state.layout.spans):
            prefix = f"node/{v}/sample/{b}/"
            fields[prefix+"times"] = state.layout.times[a:z]
            fields[prefix+"last_time"] = state.last_times[b]
            fields[prefix+"observations"] = state.observations[b]
            fields[prefix+"inert"] = state.inert[b]
            for name in FLOATS:
                fields[prefix+name] = getattr(state, name)[a:z]
    return fields


def raw_direct(anchor, result, x):
    fields = {}
    for (b, v), state in sorted(anchor.decode(result).items()):
        prefix = f"node/{v}/sample/{b}/"
        fields.update({prefix+k: t for k, t in state["slots"].items()})
        fields[prefix+"last_time"] = torch.tensor(state["last_time"], dtype=torch.int64)
        fields[prefix+"observations"] = torch.tensor(state["observations"], dtype=torch.int64)
        # StaticKV's independent KV state has implicit, identically zero inert
        # content. This literal invariant does not fill candidate state fields.
        fields[prefix+"inert"] = x.new_zeros(x.shape[-1]+1)
    return fields


def column_observations(columns, x):
    """Controls/candidate rows and Full existence from actual edge/port rows."""
    graph, size = columns.graph, x.shape[0]*x.shape[1]
    outgoing = {v: [] for v in range(len(graph.nodes))}
    for edge, message in columns.messages.items():
        outgoing[graph.edges[edge].source].append(message)
    for port, message in columns.outputs.items():
        outgoing[graph.outputs[port]].append(message)
    values, candidates, active = {}, {}, {}
    with torch.no_grad():
        for v in range(len(graph.nodes)):
            name = f"node/{v}"
            values[name] = x.new_zeros(size)
            candidates[name] = torch.zeros(size, dtype=torch.bool)
            if v in columns.nodes:
                node = columns.nodes[v]
                columns.layout.select(node.rows)
                values[name].index_copy_(0, node.rows.to(x.device), node.controls)
                candidates[name][node.rows] = True
            if not outgoing[v]:
                raise ValueError("LH checker needs an actual outgoing edge/port for every node's Full support")
            rows = outgoing[v][0].rows
            columns.layout.select(rows)
            if any(not torch.equal(rows, m.rows) for m in outgoing[v][1:]):
                raise ValueError("candidate outgoing copies disagree on actual Full support")
            active[name] = torch.zeros(size, dtype=torch.bool)
            active[name][rows] = True
    return dict(controls=values, candidates=candidates, active=active)


def direct_observations(anchor, result):
    groups = dict(controls={}, candidates={}, active={})
    for r, members in enumerate(anchor.members):
        for i, v in enumerate(members):
            for name in groups:
                groups[name][f"node/{v}"] = result["regions"][r][name][i].flatten()
    return groups


def objective(output, raw):
    """One declared task: 1e-6 Σ_port(cG)² + 1e-3 Σ_raw-float field²."""
    loss = output.square().sum()*1e-6
    for name in sorted(raw):
        value = raw[name]
        if value.is_floating_point():
            loss = loss+value.square().sum()*1e-3
    return loss


def capture(groups, budget):
    """Owned CPU observations; bound retained bytes before making the copies."""
    size = sum(t.numel()*t.element_size() for fields in groups.values() for t in fields.values() if t is not None)
    if size > budget:
        raise MemoryError(f"LH comparison capture needs {size} bytes, ceiling {budget}")
    return {group: {name: None if t is None else t.detach().cpu().clone() for name, t in fields.items()}
            for group, fields in groups.items()}, size


def evaluate(binding, spec, plan, host_x, support, *, level, client="python", library=None,
             workspace_bytes=2**30, capture_bytes=2**30, left_entries=0, start=0, root_family="task"):
    """Independent forward/VJP; shared immutable parameters, fresh input roots."""
    from tidegraph import ExecutionPlacement
    from lh_tensor_baseline import StaticKV
    from lh_tensor_observations import (ROOT_FAMILIES, initial_banks, initial_continuation,
                                       left_roots, direct_roots, column_roots, root_objective)
    if root_family not in ROOT_FAMILIES:
        raise ValueError("unknown LH root family")
    device = binding.model.nodes[0].bias.device
    x = host_x.to(device).detach().clone().requires_grad_()
    anchor = StaticKV(binding, spec)
    left = initial_banks(anchor, x, start, left_entries)
    placement = ExecutionPlacement(preset="cpu" if device.type == "cpu" else "mixed-c", scoring_dtype=str(x.dtype).split(".")[-1])
    if level == "direct":
        result = anchor(x, support=support.to(device), records=True, left=left, start=start)
        output = result["output_controls"][..., None]*result["output_contents"]
        raw = raw_direct(anchor, result, x)
        groups = direct_observations(anchor, result)
        work = dict(zip(("candidate_events", "selected_events", "visited_edges"), result["counts"].detach().cpu().tolist()))
        work["physical_full_rows"] = result["physical_full_rows"]
        execution = dict(schedule=result["implementation"], roots="numerical task projection")
        observed = direct_roots(anchor, result, root_family)
    else:
        layout, payloads = ports(x, support)
        from tidegraph.weighted.column_layout import SequenceLayout
        layout = SequenceLayout(layout.offsets, layout.times+start)
        initial = initial_continuation(anchor, left, start, x.shape[0])
        if level == "public":
            from tidegraph.weighted.settle_columns import SettleColumnPlan
            runner = SettleColumnPlan(binding, spec, implementation=client, native_library=library,
                                     placement=placement, max_workspace_bytes=workspace_bytes)
            result = columns = runner.run_ports(layout, payloads, left=runner.import_state(initial))
            execution = dict(schedule="public SettleColumnPlan", client=client, placement=runner.placement_record())
        elif level == "default":
            from tidegraph.weighted.execution import eager_runner
            runner = eager_runner(binding, client, placement, native_library=library, trace=False)
            result = runner.run_ports(spec, layout, payloads, state=initial,
                stop_position=start+plan["positions"], max_workspace_bytes=workspace_bytes)
            columns, execution = result.columns, result.execution
            if columns is None:
                raise ValueError("LH checker requested actual column execution; default reported a graph fallback")
        else:
            raise ValueError("unknown LH diagnostic level")
        output = output_values(result, x, len(binding.graph.outputs))
        raw, groups = raw_columns(columns), column_observations(columns, x)
        work = dict(result.stats)
        work["physical_full_rows"] = work["selected_events"]
        observed = column_roots(columns, root_family)
    if root_family == "raw":
        observed = raw
    loss = objective(output, raw) if root_family == "task" else root_objective(observed)
    groups.update(raw=raw, outputs={f"port/{p}": output[p] for p in range(len(binding.graph.outputs))}, task={"loss": loss})
    if root_family != "task":
        groups["root_values"] = observed
    parameters = dict(binding.model.named_parameters())
    roots = {"input": x, **left_roots(left),
             **{"parameter/"+name: p for name, p in parameters.items() if p.requires_grad}}
    gradients = (torch.autograd.grad(loss, tuple(roots.values()), allow_unused=True)
                 if loss.requires_grad else (None,)*len(roots))
    groups["gradients"] = dict(zip(roots, gradients))
    owned, size = capture(groups, capture_bytes)
    return owned, dict(execution=execution, actual_work=work, capture_bytes=size,
                       root_count=len(roots), connected_roots=sum(g is not None for g in gradients),
                       root_family=root_family, left_entries=left_entries, start_position=start)


def compare(actual, expected, *, max_differences=32):
    """Use repository strict tolerances; record projection-only root differences."""
    from tidegraph.comparison import collect_comparison, ComparisonMismatch
    results = {}
    if set(actual) != set(expected):
        raise ValueError("LH comparison observation domains differ")
    for group in actual:
        left, right = actual[group], expected[group]
        connections = []
        if group == "gradients" and set(left) == set(right):
            left, right = dict(left), dict(right)
            for name in left:
                a, e = left[name], right[name]
                if (a is None) != (e is None):
                    present = e if a is None else a
                    connections.append(dict(field=name, actual_none=a is None, expected_none=e is None,
                                            present_is_zero=not bool(torch.count_nonzero(present))))
                    if a is None:
                        left[name] = torch.zeros_like(present)
                    else:
                        right[name] = torch.zeros_like(present)
        try:
            result = collect_comparison(left, right, path=group, max_differences=max_differences)
        except ComparisonMismatch as error:
            result = dict(strict_passed=False, failure_kind=error.kind, error=error.record())
        if group == "gradients":
            result.update(connection_differences=connections, connection_equivalent=not connections,
                          scope="None-as-zero numerical task VJP; not all-root/connection equivalence")
        results[group] = result
    hard = any(r.get("failure_kind") in {"nonfinite", "structure", "discrete"} or r.get("discrete") == "failed" for r in results.values())
    numeric = any(r.get("numeric") == "failed" or r.get("failure_kind") == "numeric" for r in results.values())
    return dict(strict_numerical_projection_passed=all(r["strict_passed"] for r in results.values()),
                hard_failure=hard, numerical_failure=numeric, sections=results)
