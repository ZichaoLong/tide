"""Readonly field owners between numeric stages, with independent column roots.

A publication compacts all actual node rows once per field. Node handles are
views of that owner, not separately gathered tensors. A consumer may borrow a
contiguous span of those handles without copying, but its VJP must still target
the actual handles: returning their storage base would bypass public roots.
"""
from dataclasses import dataclass
import torch
from .packing_observation import projection


@dataclass(frozen=True)
class FieldBlock:
    """One actual-row owner; lengths and gather coordinates are integer layout."""
    value: torch.Tensor
    lengths: tuple

    @classmethod
    def publish(cls, field, indices, lengths):
        # Called inside the producer's no-grad publication boundary. Do not
        # preserve the private formula's edge as a second public parent.
        value = field.flatten(0, 2).index_select(0, indices).detach()
        projection("column_block_publish", len(lengths))
        return cls(value, tuple(lengths))

    def columns(self):
        return tuple(value.detach() for value in self.value.split(self.lengths))


def _span(values):
    """Prove an ordered exact storage span; equality of values is irrelevant."""
    first = values[0]
    if first.ndim < 1 or first.layout != torch.strided or not first.is_contiguous() or first.is_conj() or first.is_neg():
        return None
    storage, cursor = first.untyped_storage()._cdata, first.storage_offset()
    for value in values:
        if (value.ndim != first.ndim or value.shape[1:] != first.shape[1:]
                or (value.dtype, value.device) != (first.dtype, first.device)
                or value.layout != torch.strided or not value.is_contiguous() or value.is_conj() or value.is_neg()
                or value.untyped_storage()._cdata != storage or value.storage_offset() != cursor):
            return None
        cursor += value.numel()
    return (sum(len(value) for value in values), *first.shape[1:])


def _strides(shape):
    # Empty and size-one axes can be is_contiguous with arbitrary stored
    # strides. Enlarging a first slice must use the logical contiguous layout.
    result, stride = [], 1
    for size in reversed(shape):
        result.append(stride)
        stride *= max(size, 1)
    return tuple(reversed(result))


class _Borrow(torch.autograd.Function):
    @staticmethod
    def forward(ctx, shape, *values):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(*values)
        ctx.lengths = tuple(len(value) for value in values)
        return values[0].as_strided(shape, _strides(shape)).detach()

    @staticmethod
    def backward(ctx, bar):
        if torch.is_grad_enabled():
            raise ValueError("column block borrowing supports first-order VJP only")
        values = ctx.saved_tensors  # Includes all original storage versions.
        parts = (None,)*len(values) if bar is None else bar.split(ctx.lengths)
        return None, *(part if value.requires_grad else None for part, value in zip(parts, values))


def concatenate(values):
    """Join whole column roots, borrowing an actual owner when layout permits.

    This is an ordinary concatenation map, including its whole-column gradient
    support. Regional branch-aware VJPs still supply their own explicit support.
    No private graph/base edge is used in place of the supplied root edges.
    """
    values = tuple(values)
    if len(values) == 1:
        return values[0]
    if not values:
        raise ValueError("column concatenation needs at least one root")
    shape = _span(values)
    if shape is None:
        projection("column_block_copy", len(values))
        return torch.cat(values)
    projection("column_block_borrow", len(values))
    if torch.is_grad_enabled() and any(value.requires_grad for value in values):
        return _Borrow.apply(shape, *values)
    return values[0].as_strided(shape, _strides(shape)).detach()


def stack_columns(values):
    """[node, row, ...] -> [row, node, ...], without a new numerical bank."""
    values = tuple(values)
    if not values or any(value.shape != values[0].shape for value in values):
        raise ValueError("column stack requires matching nonempty column shapes")
    return concatenate(values).reshape(len(values), *values[0].shape).movedim(0, 1)
