"""CPU adapters: independent seeded inputs, loss VJP and actual-field decode.

The record roundtrip checks the observation codec using an independently
scheduled fixture. It is not device execution or resident numerical evidence.
"""
from types import SimpleNamespace
import pytest
import torch
from tidegraph import Atom, Continuation, External
from tidegraph.compare import equivalent
from tidegraph.experiment.block_inputs import BlockInputs, output_loss
from tidegraph.experiment import block_records
from tidegraph.experiment.tasks import WeightedOutputTask
from tidegraph.planning.hierarchy import general_ports_plan
from tidegraph.planning.materialize import materialize
from tidegraph.planning.model import resolve_model, workload
from tidegraph.reference import run
from tidegraph.settle import SettleGraph


def fixture(pattern):
    plan, _ = general_ports_plan(samples=3, positions=4, width=4)
    definition = {k: plan[k] for k in ("graph", "memories", "regions", "classes", "source_binding", "source_hidden_width", "alpha_agg")}
    definition["memories"] = [dict(c, memory_read="qry") for c in definition["memories"]]
    model, _ = resolve_model(**definition)
    plan, _ = workload(model, samples=3, positions=4, input_pattern=pattern)
    binding, templates, _ = materialize(plan, dtype=torch.float32, max_bytes=10**8)
    spec = SettleGraph(binding.graph, tuple(plan["graph"]["ranks"]), binding="ports", time_stride=plan["stride"])
    return plan, spec, binding.model, templates


def externals(spec, inputs, state):
    positions, payload, present = inputs
    count = {owner: x[0] for owner, x in state.ledger.items()}
    result = []
    for b, t, p in present.nonzero().tolist():
        count[b, p] = count.get((b, p), -1)+1
        result.append(External(b, p, count[b, p], int(positions[b, t])*spec.stride+spec.rank(spec.graph.inputs[p]), payload[b, t, p]))
    return result


def actual_fields(spec, inputs, expected):
    """Encode independent fixture values into public field shapes, without VJP."""
    positions, payload, present = inputs
    batch, length, _, width = payload.shape
    frames, fields, metadata = batch*length, {}, dict(positions=positions.clone())
    def field(name, shape, mask):
        fields[name] = SimpleNamespace(values=torch.zeros(shape), present=mask, valid=None)
        return fields[name].values
    first_position = int(positions[0, 0])
    for kind, count in (("edge", len(spec.graph.edges)), ("output", len(spec.graph.outputs))):
        mask = torch.zeros((frames, count), dtype=torch.bool)
        field(kind+".strength", (frames, count), mask)
        field(kind+".content", (frames, count, width-1), mask)
        metadata[kind+".times"] = torch.zeros((frames, count), dtype=torch.int64)
    for atom in expected.messages:
        frame = atom.batch*length+atom.time//spec.stride-first_position
        fields["edge.strength"].present[frame, atom.source] = True
        fields["edge.strength"].values[frame, atom.source] = atom.value[0]
        fields["edge.content"].values[frame, atom.source] = atom.value[1:]
        metadata["edge.times"][frame, atom.source] = atom.time
    for b, time, port, value in expected.outputs:
        frame = b*length+time//spec.stride-first_position
        fields["output.strength"].present[frame, port] = True
        fields["output.strength"].values[frame, port] = value[0]
        fields["output.content"].values[frame, port] = value[1:]
        metadata["output.times"][frame, port] = time
    metadata["input_counts"] = torch.full(present.sum(1).shape, -1, dtype=torch.int64)
    for owner, pair in expected.continuation.ledger.items():
        metadata["input_counts"][owner] = pair[0]
    for r in range(len(spec.graph.regions)):
        p = f"region.{r}."
        nodes = [v for v, node in enumerate(spec.graph.nodes) if node.region == r]
        n, rows = len(nodes), frames*len(nodes)
        events = [e for e in expected.trace if e["node"] in nodes]
        candidate = torch.zeros(rows, dtype=torch.bool)
        selected = torch.zeros((frames, n), dtype=torch.bool)
        metadata[p+"nodes"] = torch.tensor(nodes)
        metadata[p+"candidate_times"] = torch.zeros(rows, dtype=torch.int64)
        metadata[p+"fiber_lengths"] = torch.zeros(rows, dtype=torch.int64)
        field(p+"received", (rows, width), candidate)
        field(p+"controls", (frames, n), candidate.reshape(frames, n))
        field(p+"message.strength", (frames, n), selected)
        field(p+"message.content", (frames, n, width-1), selected)
        if not events:
            continue
        capacity = max([len(e["proposal_slots"]["times"]) for e in events]+[1])
        metadata[p+"proposal.integer"] = torch.zeros((rows, capacity+4), dtype=torch.int64)
        field(p+"proposal.value", (rows, width), candidate)
        for name, value in events[0]["proposal_slots"].items():
            if name != "times":
                field(p+"proposal."+name, (rows, capacity, *value.shape[1:]), candidate)
        for name, value in events[0]["descriptor"].items():
            field(p+"Describe."+name, (rows, *value.shape), candidate)
        for e in events:
            frame = e["batch"]*length+e["time"]//spec.stride-first_position
            local, row = nodes.index(e["node"]), frame*n+nodes.index(e["node"])
            candidate[row] = True
            metadata[p+"candidate_times"][row] = e["time"]
            metadata[p+"fiber_lengths"][row] = len(e["fiber"])
            fields[p+"received"].values[row] = e["content"]
            fields[p+"controls"].values[frame, local] = e["control"]
            fields[p+"proposal.value"].values[row] = e["proposal"]
            stamps = e["proposal_slots"]["times"]
            raw = metadata[p+"proposal.integer"][row]
            raw[:3] = torch.tensor([int(stamps[-1]) if len(stamps) else -1, len(stamps), len(stamps)])
            raw[4:4+len(stamps)] = stamps
            for name, value in e["proposal_slots"].items():
                if name != "times":
                    fields[p+"proposal."+name].values[row, :len(stamps)] = value
            for name, value in e["descriptor"].items():
                fields[p+"Describe."+name].values[row] = value
            if e["active"]:
                selected[frame, local] = True
                fields[p+"message.strength"].values[frame, local] = e["full"][0]
                fields[p+"message.content"].values[frame, local] = e["full"][1:]
    return SimpleNamespace(fields=fields, metadata=metadata)


@pytest.mark.parametrize("pattern", [None, "staggered-v1"])
def test_tensor_input_material_and_complete_candidate_decode_across_two_windows(pattern):
    plan, spec, model, templates = fixture(pattern)
    inputs, task = BlockInputs(plan), WeightedOutputTask({}, plan)
    state = Continuation(spec.graph.identity, plan["samples"])
    with torch.no_grad():
        for index in (0, 1):
            tensor = inputs.window(index)
            external = externals(spec, tensor, state)
            want = task.inputs(templates, index)
            key = lambda x: (x.batch, x.port, x.position, x.time)
            equivalent(sorted(external, key=key), sorted(want, key=key), atol=0, rtol=0)
            expected = run(spec.graph, model, state, external, (index+1)*plan["span"], sealed_until=(index+1)*plan["span"])
            window = actual_fields(spec, tensor, expected)
            stats = block_records.work(window, input_positions=inputs.input_positions,
                                       input_atoms=inputs.input_atoms)
            saved = block_records.window_snapshot(window, complete=True)
            actual = block_records.complete(saved, spec, tensor, expected.continuation.detach(), stats["actual_stats"])
            equivalent(actual, expected, atol=0, rtol=0)
            assert stats["candidate_updates"] == len(expected.trace)
            assert stats["messages"] == len(expected.messages)
            # Throughput counts logical sample/positions with an actual input,
            # not padding in the rectangular resident allocation or port copies.
            assert stats["input_positions"] == len({(e.batch, e.time//spec.stride) for e in external})
            assert stats["actual_stats"]["input_atoms"] == len(external)
            assert stats["actual_stats"]["allocated_input_positions"] == plan["samples"]*plan["positions"]
            # Missing candidates cannot be filled by a reference recurrence.
            name = next(n for n in saved["fields"] if n.endswith(".received"))
            saved["fields"][name]["present"].zero_()
            with pytest.raises(ValueError, match="without its candidate"):
                block_records.complete(saved, spec, tensor, expected.continuation.detach(), {})
            state = expected.continuation.detach()


@pytest.mark.parametrize("scale", [0., .17])
def test_consumer_field_loss_and_connected_zero_cotangents(scale):
    c = torch.tensor([[.5, .25], [0., 3.]], requires_grad=True)
    g = torch.tensor([[[2., -1.], [3., 4.]], [[7., 9.], [1., 2.]]], requires_grad=True)
    live = torch.tensor([[True, True], [True, False]])
    fields = {"output.strength": SimpleNamespace(values=c.detach(), present=live),
              "output.content": SimpleNamespace(values=g.detach(), present=live)}
    with torch.no_grad():
        loss, roots = output_loss(fields, scale, training=True)
    independent = sum((c[b, p]*g[b, p]).square().sum()*scale for b, p in live.nonzero().tolist())
    bars = torch.autograd.grad(independent, (c, g))
    torch.testing.assert_close(loss, independent)
    for name, expected in zip(("output.strength", "output.content"), bars):
        torch.testing.assert_close(roots[name], expected)
        assert not roots[name].requires_grad


def test_zero_mass_is_a_present_candidate_and_keeps_exact_old_cache():
    plan, spec, model, _ = fixture(None)
    source = BlockInputs(plan)
    state = Continuation(spec.graph.identity, plan["samples"])
    with torch.no_grad():
        for index in (0, 1):
            positions, payload, present = source.window(index)
            payload = payload.clone()
            if index:
                payload[..., 0] = 0.
            inputs = positions, payload, present
            expected = run(spec.graph, model, state, externals(spec, inputs, state),
                (index+1)*plan["span"], sealed_until=(index+1)*plan["span"])
            window = actual_fields(spec, inputs, expected)
            saved = block_records.window_snapshot(window, complete=True)
            actual = block_records.complete(saved, spec, inputs, expected.continuation.detach(), {})
            equivalent(actual, expected, atol=0, rtol=0)
            if index:
                assert actual.trace and not actual.outputs and all(not e["active"] for e in actual.trace)
                for event in actual.trace:
                    previous = state.states[event["batch"], event["node"]]
                    equivalent(event["proposal_slots"], previous.slots, atol=0, rtol=0)
            state = expected.continuation.detach()
