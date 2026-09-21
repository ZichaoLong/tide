"""Associative affine prefix scan: y[t] = a[t] * y[t-1] + b[t]."""
import torch


def affine_scan(a, b, initial):
    if len(b) == 0:
        return b
    stride = 1
    while stride < len(b):
        b = torch.cat((b[:stride], b[stride:] + a[stride:] * b[:-stride]))
        a = torch.cat((a[:stride], a[stride:] * a[:-stride]))
        stride *= 2
    return b + a * initial
