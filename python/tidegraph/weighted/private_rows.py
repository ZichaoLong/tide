"""Packed private autograd leaves; original rows remain independent outer roots.

Only structurally homogeneous query/region groups may use this boundary.
Undefined cotangent masks are still evaluated separately, and only participating
rows receive sliced gradients. Original SavedVariables keep version guards.
"""
import math
import torch


def join(values, *, stack=False):
    """Collect actual views by one interval or gather, never per-view transposes."""
    if len(values) > 1:
        base = values[0]._base
        shapes_match = (all(t.shape == values[0].shape for t in values) if stack else
                        all(t.shape[1:] == values[0].shape[1:] for t in values))
        if (base is not None and base.is_contiguous() and shapes_match
                and all(t._base is base and t.is_contiguous() and not t.is_conj() and not t.is_neg()
                        and t.requires_grad == base.requires_grad for t in values)):
            shape = ((len(values), *values[0].shape) if stack else
                     (sum(t.shape[0] for t in values), *values[0].shape[1:]))
            cursor, contiguous = values[0].storage_offset(), True
            for value in values:
                contiguous = contiguous and value.storage_offset() == cursor
                cursor += value.numel()
            if contiguous:
                return base.reshape(-1).narrow(0, values[0].storage_offset()-base.storage_offset(),
                                               sum(t.numel() for t in values)).reshape(shape)
            unit = values[0].numel() if stack else math.prod(values[0].shape[1:])
            if unit and base.numel() % unit == 0 and all((t.storage_offset()-base.storage_offset()) % unit == 0
                                                        and t.numel() % unit == 0 for t in values):
                indices = []
                for value in values:
                    start = (value.storage_offset()-base.storage_offset())//unit
                    indices.extend(range(start, start+value.numel()//unit))
                ids = torch.tensor(indices, dtype=torch.int64, device=base.device)
                return base.reshape(-1, unit).index_select(0, ids).reshape(shape)
    return torch.stack(values) if stack else torch.cat(values)


class PrivateRows:
    def __init__(self, inputs, weights, rows):
        self.weights, self.rows = weights, rows
        self.fields = (len(inputs)-weights)//rows
        self.spans = []
        self.shapes = [t.shape for t in inputs[weights:]]
        self.leaves = [t.detach().requires_grad_(t.requires_grad) for t in inputs[:weights]]
        for field in range(self.fields):
            column = [inputs[weights+row*self.fields+field] for row in range(rows)]
            assert len({(t.requires_grad, t.dtype, t.device) for t in column}) == 1
            offsets, cursor = [], 0
            for t in column:
                offsets.append((cursor, t.numel()))
                cursor += t.numel()
            self.spans.append(offsets)
            with torch.no_grad():
                packed = column[0].detach().reshape(-1) if rows == 1 else torch.cat([t.detach().reshape(-1) for t in column])
            self.leaves.append(packed.requires_grad_(column[0].requires_grad))

    def views(self):
        # Called with grad enabled: join() can later remove these view transposes.
        result = list(self.leaves[:self.weights])
        for row in range(self.rows):
            for field in range(self.fields):
                start, length = self.spans[field][row]
                result.append(self.leaves[self.weights+field].narrow(0, start, length).reshape(
                    self.shapes[row*self.fields+field]))
        return result

    def scatter(self, leaf_id, used, gradient, result):
        if gradient is None:
            return
        if leaf_id < self.weights:
            old = result[leaf_id]
            result[leaf_id] = gradient if old is None else old+gradient
        else:
            field = leaf_id-self.weights
            for row in used:
                index = self.weights+row*self.fields+field
                start, length = self.spans[field][row]
                value = gradient.narrow(0, start, length).reshape(self.shapes[row*self.fields+field])
                result[index] = value if result[index] is None else result[index]+value
