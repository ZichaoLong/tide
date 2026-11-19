"""Small literal KV-extension witnesses, independent of production math helpers.

This is deliberately a bounded scalar/node loop, not an alternative evaluator.
Content/summary/FFN widths are independent of the total Q/K/V channel width.
"""
import math

import torch
from torch.nn import functional as F

from tide.kv import KVMemory
from scripts.qualify_precision import THRESHOLDS, comparison, fingerprint


def dense_matrix(layer, owner):
    base = layer.weight[int(layer.groups[owner])]
    return base + layer.left[owner] @ layer.right[owner] if layer.shared else base


def normalized(x):
    return x / torch.sqrt(x.square().mean(-1, keepdim=True) + 1e-5)


def cache_contents(state):
    return [[{int(state.times[n, b, i]): tuple(value[n, b, i] for value in
              (state.weight, state.keys, state.values, state.features))
              for i in range(state.times.shape[2]) if bool(state.weight[n, b, i] > 0)}
             for b in range(state.times.shape[1])] for n in range(state.times.shape[0])]


def direct_formula(module, u, mass, times, owners, control):
    """Finite no-initial-state Read+ and Full, including true zero-write VJP."""
    descriptions, outputs, caches = [], [], []
    frequencies = u.new_tensor([10000 ** (-2 * i / 32) for i in range(16)])
    for n in range(u.shape[0]):
        owner = int(owners[n])
        z = normalized(u[n])
        keys, values = (z @ dense_matrix(module.kv, owner).T).chunk(2, -1)
        features = z @ dense_matrix(module.summary, owner).T
        # Absence is a branch with zero derivative, rather than a zero key prior.
        weight = torch.where(mass[n] > 0, mass[n], torch.zeros_like(mass[n]))
        delta = times[n, :, :, None] - times[n, :, None, :]
        visible = (delta >= 0) & (delta < module.window)
        lw = weight[:, None] * visible
        sw = lw * torch.exp(delta.clamp_min(0).to(u.dtype) * -module.short_decay)
        ws, wl = sw.sum(-1), lw.sum(-1)
        short = sw @ features / torch.where(ws > 1, ws, 1)[..., None]
        long = lw @ features / torch.where(wl > 1, wl, 1)[..., None]
        descriptions.append((features, short, long, ws, wl))
        node_outputs, node_caches = [], []
        for b in range(u.shape[1]):
            rows = []
            for i in range(u.shape[2]):
                if not bool(control[n, b, i] > 0):
                    rows.append(torch.zeros_like(u[n, b, i]))
                    continue
                keep = visible[b, i] & (weight[b] > 0)
                if not bool(keep.any()):
                    raise ValueError("formula fixture requires selected queries to see a positive write")
                key = keys[b, keep].reshape(-1, module.heads, 16, 2)
                # Subtract original int64 times before any floating conversion.
                relative = (times[n, b, keep] - times[n, b, i]).to(u.dtype) / module.position_period
                phase = relative[:, None] * frequencies
                rotated = torch.stack((key[..., 0] * phase.cos()[:, None] - key[..., 1] * phase.sin()[:, None],
                                       key[..., 0] * phase.sin()[:, None] + key[..., 1] * phase.cos()[:, None]), -1).flatten(-2)
                query = torch.stack([dense_matrix(head, owner) @ normalized(u[n, b, i])
                                     for head in module.query.heads])
                logits = (rotated * query).sum(-1).T / math.sqrt(32)
                terms = weight[b, keep][None] * torch.exp(logits - logits.amax(-1, keepdim=True))
                probability = terms / terms.sum(-1, keepdim=True)
                read = (probability[..., None] * values[b, keep].reshape(-1, module.heads, 32).transpose(0, 1)).sum(1)
                amount = weight[b, keep].sum().clamp(max=1)
                x = u[n, b, i] + dense_matrix(module.output, owner) @ (read.flatten() * amount)
                zfull = normalized(x)
                hidden = F.silu(dense_matrix(module.ffn.gate, owner) @ zfull) * (dense_matrix(module.ffn.up, owner) @ zfull)
                rows.append(x + dense_matrix(module.ffn.down, owner) @ hidden)
            node_outputs.append(torch.stack(rows))
            positive = weight[b] > 0
            last = times[n, b, positive].max() if bool(positive.any()) else times.new_tensor(0)
            retained = positive & ((last - times[n, b]) < module.window)
            node_caches.append({int(times[n, b, i]): (weight[b, i], keys[b, i], values[b, i], features[b, i])
                                for i in retained.nonzero().flatten().tolist()})
        outputs.append(torch.stack(node_outputs))
        caches.append(node_caches)
    return dict(description=tuple(torch.stack([x[i] for x in descriptions]) for i in range(5)),
                full=torch.stack(outputs), state=caches)


def projected(module, u, mass, times, owners, control, *, cut=None):
    state, outputs, descriptions = None, [], []
    starts = (0, u.shape[2]) if cut is None else (0, cut, u.shape[2])
    for lo, hi in zip(starts[:-1], starts[1:]):
        desc, context, state = module.prepare(u[:, :, lo:hi], mass[:, :, lo:hi], times[:, :, lo:hi], owners, state)
        outputs.append(module.finish(context, control[:, :, lo:hi]))
        descriptions.append(desc)
    return dict(description=tuple(torch.cat([row[i] for row in descriptions], 2) for i in range(5)),
                full=torch.cat(outputs, 2), state=cache_contents(state))


def cotangent_loss(result):
    from scripts.qualify_precision import tensors
    losses = []
    for index, (_, value) in enumerate(tensors(result)):
        cotangent = torch.sin(torch.arange(value.numel(), dtype=torch.float32, device=value.device) + index)
        losses.append((value * cotangent.reshape(value.shape).to(value.dtype)).mean())
    return sum(losses)


def fixture(width=32, *, device="cpu", dtype=torch.float32, shared=True):
    torch.manual_seed(2419)
    module = KVMemory(3, 192, [0, 1, 0] if shared else None, 8, 4,
                      backend="math", chunk=3, window=16, kv_width=width).to(device=device, dtype=dtype)
    u = torch.randn(2, 2, 7, 192, dtype=dtype, device=device, requires_grad=True)
    mass = torch.tensor([[[.2, 0, .3, .4, 0, .2, .3], [.1, .2, 0, .3, .2, 0, .1]],
                         [[.3, .2, 0, .1, .3, .1, 0], [0, .3, .2, .1, 0, .3, .2]]],
                        dtype=dtype, device=device, requires_grad=True)
    times = (2**55 + 19 + torch.arange(7, device=device, dtype=torch.int64) * 4).expand(2, 2, 7)
    owners = torch.tensor([2, 0], device=device)
    control = torch.zeros_like(mass)
    control[:, :, 2::2] = (mass[:, :, 2::2] > 0).to(dtype)
    return (module, u, mass, times, owners, control), (u, mass, *module.parameters())


def formula_witness(width=32, *, device="cpu", dtype=torch.float32):
    args, roots = fixture(width, device=device, dtype=dtype)
    actual, expected = projected(*args), direct_formula(*args)
    chunked = projected(*args, cut=4)
    named = ["input", "mass", *(name for name, _ in args[0].named_parameters())]
    gradients = []
    for result in (actual, expected, chunked):
        gradients.append(dict(zip(named, torch.autograd.grad(cotangent_loss(result), roots, retain_graph=True))))
    precision = "fp64" if dtype == torch.float64 else "fp32"
    checks = {"value_formula": comparison(actual, expected, THRESHOLDS[precision]),
              "vjp_formula": comparison(gradients[0], gradients[1], THRESHOLDS[precision]),
              "value_continuation": comparison(chunked, actual, THRESHOLDS[precision]),
              "vjp_continuation": comparison(gradients[2], gradients[0], THRESHOLDS[precision])}
    zero = [int(torch.count_nonzero(row["mass"][args[2] == 0])) for row in gradients]
    return dict(width=width, d=192, summary_width=48, ffn_width=768, heads=width // 32,
                head_width=32, state="nonfinite" if any(c["nonfinite"] for c in checks.values()) else
                "structural_failed" if any(c["metadata_failures"] for c in checks.values()) or any(zero) else
                "finite_deviation" if any(c["threshold_failures"] for c in checks.values()) else "within_threshold",
                scope="FP32 or bounded CPU FP64 literal formula, private adapters and node state, int64 translation, zero write, window eviction, full VJP",
                identity=dict(parameters=fingerprint(args[0].state_dict()), input=fingerprint(args[1]),
                              mass=fingerprint(args[2]), times=fingerprint(args[3])),
                zero_mass_gradient_nonzeros=zero, first_time=int(args[3][0, 0, 0]),
                retained_times=[[list(bank) for bank in row] for row in actual["state"]], checks=checks)
