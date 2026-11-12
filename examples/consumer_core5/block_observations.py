"""Independent scalar records check the installed owner's actual raw decoder.

This is consumer validation, not a source of candidate state or scheduling.
"""
import torch


PREFIX_ROOTS = {"keys": .017, "values": .019}


def reference_prefixes(result, continuation, model):
    states = {key: dict(value=state.value, **state.slots)
              for key, state in continuation.states.items()}
    expected = {}
    for event in result.trace:
        b, node, time = (event[name] for name in ("batch", "node", "time"))
        if (b, node) not in states:
            initial = model.nodes[node].initial()
            states[b, node] = dict(value=initial.value, **initial.slots)
        proposal = dict(value=event["proposal"], **event["proposal_slots"])
        expected[b, node, time] = (states[b, node], proposal)
        states[b, node] = dict(value=event["next"], **event["next_slots"])
    return expected


def prefix_loss(expected):
    return sum(value[name].sum()*scale for _, value in expected.values()
               for name, scale in PREFIX_ROOTS.items())


def check_prefixes(session, window, spec, expected):
    """Materialize actual raw values; roots also work before this observation."""
    positions = window.metadata["positions"].cpu()
    length = positions.shape[1]
    observed = session.observe(window)
    roots = {}
    for r in range(len(spec.graph.regions)):
        nodes = [n for n, node in enumerate(spec.graph.nodes) if node.region == r]
        for label in ("old", "proposal"):
            for name in ("value", "keys", "values", "summaries", "weights"):
                key = f"region.{r}.{label}.{name}"
                field, actual = window.fields[key], observed[key]
                assert field.values is None, "advance must not expand raw prefixes"
                assert list(actual.values.shape) == list(field.shape)
                assert torch.equal(actual.present, field.present)
                data = actual.values.cpu()
                flags = actual.present.cpu()
                valid = actual.valid.cpu() if actual.valid is not None else None
                for (b, node, time), pair in expected.items():
                    if node not in nodes:
                        continue
                    original = (time-spec.rank(node))//spec.stride
                    column = (positions[b] == original).nonzero().item()
                    row = (b*length+column)*len(nodes)+nodes.index(node)
                    assert bool(flags[row])
                    value = pair[label == "proposal"][name]
                    actual_value = data[row] if valid is None else data[row][valid[row]]
                    torch.testing.assert_close(actual_value, value, rtol=8e-5, atol=8e-6,
                                               msg=f"{key} batch={b} time={time}")
                if label == "proposal" and name in PREFIX_ROOTS:
                    roots[key] = torch.full(tuple(field.shape), PREFIX_ROOTS[name],
                                            dtype=torch.float32, device=field.present.device)
    return roots
