"""Ordered consumer adjoints for an actually shared Aggregate column.

Forward readers borrow views, not copies. Each node gets its own use slot so
cross-device backward arrival order cannot choose the sum of shared uses.
This is a first-order column boundary, not a new independent event-root API.
"""
import torch


class _Uses(torch.autograd.Function):
    @staticmethod
    def forward(ctx, count, *values):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(*values)
        ctx.count = count
        outputs = tuple(v.view_as(v) for _ in range(count) for v in values)
        ctx.mark_non_differentiable(*(out for i, out in enumerate(outputs)
                                     if not values[i % len(values)].requires_grad))
        return outputs

    @staticmethod
    def backward(ctx, *bars):
        if torch.is_grad_enabled():
            raise ValueError("shared column uses support first-order VJP only")
        values = ctx.saved_tensors  # Original versions, including frozen fields.
        result = [None]
        for i in range(len(values)):
            total = None
            if ctx.needs_input_grad[i+1]:
                for j in range(ctx.count):
                    bar = bars[j*len(values)+i]
                    if bar is not None:
                        total = bar if total is None else total+bar
            result.append(total)
        return tuple(result)


def shared_uses(received):
    """Keep one real aggregation and assign stable node-order use views.

    Only identical cached result objects are shared. Equal values, shapes or
    source classes do not license reuse. Empty receivers remain absent.
    """
    groups = {}
    for node, value in received.items():
        if value is not None:
            groups.setdefault(id(value), []).append(node)
    result = dict(received)
    for nodes in groups.values():
        rows, mass, content = received[nodes[0]]
        if len(nodes) < 2 or not torch.is_grad_enabled() or not (mass.requires_grad or content.requires_grad):
            continue
        uses = _Uses.apply(len(nodes), mass, content)
        for i, node in enumerate(nodes):
            result[node] = rows, uses[2*i], uses[2*i+1]
    return result
