"""Explicit LH token clock adapter; no executor calls or synthetic source rows."""
from .history import int64, increment
from .records import External


def token_inputs(outputs, continuation, layers, stop, *, body_cut, port=0):
    """Map complete body outputs [cut*layers, stop*layers) to readout token events.

    The caller seals the body interval and provides every output on the selected
    port. A globally empty token is outside original Pronounce's input domain.
    The continuation belongs to the readout graph; this function never mutates it.
    """
    q = continuation
    if (any(not int64(x) for x in (layers, stop, body_cut, port, q.cut, q.batch_size))
            or layers < 1 or q.batch_size < 1 or not 0 <= q.cut <= stop or port < 0
            or stop > (2**63-1)//layers or body_cut != stop*layers):
        raise ValueError("invalid or unsealed token window")
    selected, seen, tokens = [], set(), set()
    for batch, time, output_port, value in outputs:
        if not int64(output_port) or output_port < 0:
            raise ValueError("invalid body output port")
        if output_port != port:
            continue
        if (not int64(batch) or not int64(time) or not 0 <= batch < q.batch_size
                or not q.cut*layers <= time < body_cut or (batch, time) in seen):
            raise ValueError("invalid or duplicate body output coordinate")
        seen.add((batch, time)); token, phase = divmod(time, layers)
        selected.append((token, batch, phase, value)); tokens.add(token)
    if len(tokens) != stop-q.cut:
        raise ValueError("original Pronounce requires a nonempty global token window")
    positions = {}
    result = []
    for token, batch, phase, value in sorted(selected, key=lambda row: row[:3]):
        owner = batch, phase
        if owner not in positions:
            last = q.ledger.get(owner)
            if last is not None and (not isinstance(last, (tuple, list)) or len(last) != 2 or any(not int64(x) for x in last)
                                     or last[0] < 0 or not 0 <= last[1] < q.cut):
                raise ValueError("invalid token input ledger")
            positions[owner] = -1 if last is None else last[0]
        position = 0 if positions[owner] == -1 else increment(positions[owner])
        result.append(External(batch, phase, position, token, value))
        positions[owner] = position
    return result
