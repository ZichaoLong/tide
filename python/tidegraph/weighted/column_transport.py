"""Whole-column eager transport; numerical tensors retain their original roots."""
from dataclasses import replace
from .placement_tensors import mapper


def message(value, like, cache=None):
    if (value.content.device, value.content.dtype) == (like.device, like.dtype):
        return value
    key = id(value), like.device, like.dtype
    if cache is not None and key in cache:
        return cache[key]
    copied = mapper((value.strength, value.content), like.device, like.dtype)
    result = replace(value, strength=copied(value.strength), content=copied(value.content))
    if cache is not None:
        cache[key] = result
    return result


def fields(values, device):
    """Complete descriptors reach the region owner before joint allocation."""
    copied = mapper(values.values(), device, values["w"].dtype)
    return {k: copied(v) for k, v in values.items()}
