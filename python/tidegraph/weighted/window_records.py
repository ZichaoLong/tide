"""Event materialization only for the general-engine tensor-window fallback."""
import torch
from .tensors import Payload
from .message_columns import MessageColumns


def external(spec, layout, ports, state):
    records = []
    owners = layout.owners()
    for p, payload in ports.items():
        for sample in range(layout.samples):
            ids = (owners[payload.rows] == sample).nonzero().flatten()
            positions = layout.times[payload.rows[ids]].tolist()
            count = state.ledger.get((sample, p), (-1, -1))[0]+1
            values = Payload(payload.strength.index_select(0, ids.to(payload.strength.device)),
                             payload.content.index_select(0, ids.to(payload.content.device))).encode()
            records.extend(spec.port_external(values, positions, port=p, batch=sample, start_occurrence=count))
    return records


def outputs(spec, layout, records, like):
    locations = {(b, int(t)): a+j for b, (a, z) in enumerate(layout.spans)
                 for j, t in enumerate(layout.times[a:z])}
    grouped = {p: [] for p in range(len(spec.graph.outputs))}
    for b, time, p, value in records:
        position = (time-spec.rank(spec.graph.outputs[p]))//spec.stride
        grouped[p].append((locations[b, position], value))
    result = {}
    for p, values in grouped.items():
        values.sort(key=lambda x: x[0])
        rows = torch.tensor([r for r, _ in values], dtype=torch.int64)
        packed = torch.stack([x for _, x in values]) if values else like.new_empty((0, len(like)))
        result[p] = MessageColumns(rows, packed[:, 0], packed[:, 1:])
    return result
