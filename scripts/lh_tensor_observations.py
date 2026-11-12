"""Independent LH diagnostic material and separately rooted observations.

Left banks are literal input material, never a reference trajectory. Candidate
observations only read its actual tensor columns. Dense-anchor connection
differences remain explicit numerical projections in lh_tensor_check.
"""
import torch
from lh_tensor_state import FLOATS


ROOT_FAMILIES = ("task", "aggregate", "descriptors", "control", "full", "messages", "raw")


def initial_banks(anchor, x, start, entries):
    """Deterministic nonempty raw inputs with holes and original int64 clocks."""
    if type(entries) is not int or entries < 0 or entries > 16:
        raise ValueError("LH literal left entries must be an integer in [0,16]")
    if not entries:
        return None
    if type(start) is not int or start < 2*entries:
        raise ValueError("LH nonempty left material needs start >= 2*entries")
    result = {}
    for r, members in enumerate(anchor.members):
        c = anchor.binding.model.nodes[members[0]].kernel.config
        n, b = len(members), x.shape[0]
        times = ((start-2*torch.arange(entries, 0, -1, dtype=torch.int64, device=x.device))
                 *anchor.spec.stride+anchor.spec.ranks[r])
        ids = torch.arange(n*b*entries, device=x.device).reshape(n, b, entries)
        valid = ids.remainder(5) != 3
        valid[..., 0] = True
        bank = dict(times=times, valid=valid,
                    last_times=times.expand_as(valid).masked_fill(~valid, -1).amax(-1),
                    observations=valid.sum(-1)+3)
        for role, width in (("keys", c.width), ("values", c.width), ("summaries", c.summary_width)):
            shape = (n, b, entries, width)
            values = torch.arange(n*b*entries*width, device=x.device, dtype=x.dtype).reshape(shape)
            bank[role] = (.07*torch.sin(values*.17+r*.23+len(role))).requires_grad_()
        bank["weights"] = (.3+.05*torch.cos(ids.to(x.dtype)*.13+r)).requires_grad_()
        result[r] = bank
    return result


def initial_continuation(anchor, banks, start, samples):
    """Literal raw-field packing for a common initial state; no model evaluation."""
    from tidegraph.records import Continuation, State
    q = Continuation(anchor.binding.graph.identity, samples, cut=start*anchor.spec.stride)
    if banks is None:
        return q
    for r, members in enumerate(anchor.members):
        bank = banks[r]
        for i, v in enumerate(members):
            for b in range(samples):
                mask = bank["valid"][i, b]
                slots = dict(times=bank["times"][mask],
                             **{name: bank[name][i, b][mask] for name in FLOATS})
                q.states[b, v] = State(torch.zeros_like(anchor.binding.model.nodes[v].bias),
                    int(bank["last_times"][i, b]), int(bank["observations"][i, b]), slots)
    return q


def left_roots(banks):
    return {} if banks is None else {f"left/{r}/{name}": bank[name]
        for r, bank in sorted(banks.items()) for name in FLOATS}


def direct_roots(anchor, result, family):
    """Unpadded numerical observations, with one identity per physical use."""
    values, node_full = {}, {}
    for r, members in enumerate(anchor.members):
        record = result["regions"][r]
        for i, v in enumerate(members):
            candidate, active = record["candidates"][i], record["active"][i]
            prefix = f"node/{v}/"
            present = bool(candidate.any())
            if family == "aggregate" and present:
                values[prefix+"mass"] = record["mass"][i][candidate]
                values[prefix+"content"] = record["content"][i][candidate]
            elif family == "descriptors" and present:
                values.update({prefix+name: value[i][candidate]
                               for name, value in record["descriptors"].items()})
            elif family == "control" and present:
                values[prefix+"control"] = record["controls"][i][candidate]
            elif family == "full" and present:
                values[prefix+"G"] = record["full"][i][active]
            node_full[v] = (record["controls"][i][active], record["full"][i][active])
    if family == "messages":
        uses = [(f"edge/{e}", edge.source) for e, edge in enumerate(anchor.binding.graph.edges)]
        uses += [(f"port/{p}", v) for p, v in enumerate(anchor.binding.graph.outputs)]
        for name, v in uses:
            values[name+"/c"], values[name+"/G"] = node_full[v]
    return values


def column_roots(columns, family):
    values = {}
    if family == "messages":
        for kind, messages in (("edge", columns.messages), ("port", columns.outputs)):
            for index, message in sorted(messages.items()):
                values[f"{kind}/{index}/c"] = message.strength
                values[f"{kind}/{index}/G"] = message.content
        return values
    outgoing = {}
    for e, message in columns.messages.items():
        outgoing.setdefault(columns.graph.edges[e].source, message)
    for p, message in columns.outputs.items():
        outgoing.setdefault(columns.graph.outputs[p], message)
    for v, node in sorted(columns.nodes.items()):
        prefix = f"node/{v}/"
        if family == "aggregate":
            values[prefix+"mass"], values[prefix+"content"] = node.window.mass, node.window.content
        elif family == "descriptors":
            values.update({prefix+name: value for name, value in node.descriptors.items()})
        elif family == "control":
            values[prefix+"control"] = node.controls
        elif family == "full":
            if v not in outgoing:
                raise ValueError("LH Full root observation requires an actual outgoing edge or port")
            values[prefix+"G"] = outgoing[v].content
    return values


def root_objective(values):
    """Distinct signed cotangents per field; includes zero cotangent elements."""
    terms = []
    for i, (_, value) in enumerate(sorted(values.items())):
        if not value.is_floating_point():
            continue
        ids = torch.arange(value.numel(), device=value.device, dtype=value.dtype)
        cotangent = .031*torch.sin(ids*.13+i*.7)
        cotangent = torch.where(ids.remainder(7) == 0, 0., cotangent).reshape(value.shape)
        terms.append((value*cotangent).sum())
    if not terms:
        raise ValueError("LH root family has no floating observation")
    return sum(terms)
