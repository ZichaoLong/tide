"""Decode actual resident block observations, never execute a local formula.

The weighted binding has empty Aggregate contributions, singleton history,
adopted proposals and broadcasts on each declared local output slot. These are
checked owner capabilities; raw state, candidate times, payloads and support all
come from the candidate. Record expansion runs only at an observation boundary.
"""
from collections import defaultdict
import torch
from ..records import Atom, Result, State
from ..history import History
from .returns import snapshot


def window_snapshot(window, *, complete):
    fields = {name: dict(values=f.values, present=f.present, valid=f.valid)
              for name, f in window.fields.items() if complete or name.startswith("output.")}
    metadata = {name: x for name, x in window.metadata.items()
                if complete or name in {"positions", "output.times"}}
    return snapshot(dict(fields=fields, metadata=metadata), packing="packed")


def work(window, *, input_positions, input_atoms):
    region = sorted(name[:-len("received")] for name in window.fields if name.endswith(".received"))
    candidates = sum(window.fields[p+"received"].present.sum() for p in region)
    fulls = sum(window.fields[p+"message.strength"].present.sum() for p in region)
    received = sum(window.metadata[p+"fiber_lengths"].sum() for p in region)
    edge, output = (window.fields[p+"strength"].present.sum() for p in ("edge.", "output."))
    values = torch.stack((candidates, fulls, received, edge, output)).cpu().tolist()
    candidates, fulls, received, edge, output = map(int, values)
    stats = dict(candidate_events=candidates, selected_events=fulls, visited_edges=edge,
                 received_atoms=received, resident_regions=len(region),
                 input_atoms=input_atoms,
                 allocated_input_positions=window.metadata["positions"].numel(),
                 allocated_candidate_rows=sum(window.fields[p+"received"].values.shape[0] for p in region),
                 allocated_message_rows=sum(window.fields[p+"message.strength"].values.numel() for p in region))
    return dict(candidate_updates=candidates, full_calls=fulls, messages=edge, outputs=output,
                pending=0, actual_stats=stats, input_positions=input_positions,
                effective_window_execution=dict(schedule="settle-block-v2", input="B×L×P",
                    readiness="device", selection="device", progression="device"))


def _payload(fields, prefix):
    return torch.cat((fields[prefix+"strength"]["values"][..., None], fields[prefix+"content"]["values"]), -1)


def output_records(value, spec):
    fields, meta = value["fields"], value["metadata"]
    length = meta["positions"].shape[1]
    payload, mask, times = _payload(fields, "output."), fields["output.strength"]["present"], meta["output.times"]
    out = [(frame//length, int(times[frame, p]), p, payload[frame, p].clone())
           for frame, p in mask.nonzero().tolist()]
    return sorted(out, key=lambda x: (x[1], x[0], spec.graph.outputs[x[2]], x[2]))


def _raw(fields, meta, prefix, row):
    integer = meta[prefix+"integer"][row]
    time, observations, length, reserved = integer[:4].tolist()
    if reserved != 0 or not 0 <= length <= integer.numel()-4:
        raise ValueError("invalid actual KV raw metadata at observation boundary")
    slots = {name: fields[prefix+name]["values"][row, :length].clone()
             for name in ("keys", "values", "summaries", "weights")}
    slots["times"] = integer[4:4+length].clone()
    return State(fields[prefix+"value"]["values"][row].clone(), time, observations, slots)


def complete(value, spec, inputs, continuation, stats):
    fields, meta = value["fields"], value["metadata"]
    positions, payload, present = inputs
    length, graph = positions.shape[1], spec.graph
    fibers, messages = defaultdict(list), []
    # The final ledger contains last occurrences, not counts. Preserve missing
    # physical ports and staggered per-port numbering across earlier windows.
    occurrence = meta["input_counts"]-present.sum(1)
    for b, t, p in present.nonzero().tolist():
        occurrence[b, p] += 1
        time = int(positions[b, t])*spec.stride+spec.rank(graph.inputs[p])
        atom = Atom(b, graph.inputs[p], time, 0, p, int(occurrence[b, p]), payload[b, t, p].clone())
        fibers[b, atom.node, atom.time].append(atom)
    physical = _payload(fields, "edge.")
    for frame, edge in fields["edge.strength"]["present"].nonzero().tolist():
        e = graph.edges[edge]
        time = int(meta["edge.times"][frame, edge])
        atom = Atom(frame//length, e.target, time, 1, edge, time-e.delay, physical[frame, edge].clone())
        messages.append(atom)
        fibers[atom.batch, atom.node, atom.time].append(atom)
    trace = []
    for r in range(len(graph.regions)):
        prefix = f"region.{r}."
        nodes = meta[prefix+"nodes"].tolist()
        count = len(nodes)
        controls = fields[prefix+"controls"]["values"].reshape(-1)
        active = fields[prefix+"message.strength"]["present"].reshape(-1)
        emitted = _payload(fields, prefix+"message.").reshape(-1, payload.shape[-1])
        for row in fields[prefix+"received"]["present"].nonzero().flatten().tolist():
            frame, n = divmod(row, count)
            batch, node = frame//length, nodes[n]
            time = int(meta[prefix+"candidate_times"].reshape(-1)[row])
            fiber = sorted(fibers.pop((batch, node, time), []), key=lambda a: a.key())
            if not fiber or len(fiber) != int(meta[prefix+"fiber_lengths"].reshape(-1)[row]):
                raise ValueError("actual block candidate/fiber count differs")
            state = _raw(fields, meta, prefix+"proposal.", row)
            descriptor = {}
            for name, f in fields.items():
                if name.startswith(prefix+"Describe."):
                    key = name[len(prefix+"Describe."):]
                    x = f["values"][row].clone()
                    descriptor[key] = x.reshape(()) if key in {"w", "mu", "score"} else x
            event = dict(batch=batch, node=node, time=time, fiber=fiber,
                content=fields[prefix+"received"]["values"][row].clone(), contributions={},
                proposal=state.value, descriptor=descriptor, active=bool(active[row]), control=controls[row].clone(),
                comparison=state.value.clone(), next=state.value.clone(), history=History(last_time=time),
                proposal_slots=state.slots, comparison_slots=snapshot(state.slots), next_slots=snapshot(state.slots))
            if event["active"]:
                offsets = graph.port_indexes[1].offsets
                event["full"] = emitted[row].clone()
                event["emitted"] = {slot: emitted[row].clone() for slot in range(offsets[node+1]-offsets[node])}
            trace.append(event)
    if fibers:
        raise ValueError("actual block leaves an incoming atom without its candidate")
    trace.sort(key=lambda e: (e["time"], e["batch"], e["node"]))
    messages.sort(key=lambda a: (a.position, a.batch, graph.edges[a.source].source, a.source))
    return Result(continuation, trace, output_records(value, spec), messages, dict(stats))
