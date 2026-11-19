"""Small literal KV-extension witnesses, independent of production math helpers.

This is deliberately a bounded scalar/node loop, not an alternative evaluator.
Content/summary/FFN widths are independent of the total Q/K/V channel width.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

import torch
from torch.nn import functional as F

from tide.kv import KVMemory
from tide.precision import precision_policy
from benchmarks.statistics import RuntimeAudit
from scripts.job import atomic
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


def fixture(width=32, *, d=192, device="cpu", dtype=torch.float32, shared=True, batch=2):
    if type(d) is not int or d < 32 or d % 32:
        raise ValueError("content width must be a positive multiple of 32")
    if type(batch) is not int or batch not in (1, 2):
        raise ValueError("this finite fixture supports batch 1 or 2")
    torch.manual_seed(2419)
    module = KVMemory(3, d, [0, 1, 0] if shared else None, 8, 4,
                      backend="math", chunk=3, window=16, kv_width=width).to(device=device, dtype=dtype)
    u = torch.randn(2, batch, 7, d, dtype=dtype, device=device, requires_grad=True)
    mass = torch.tensor([[[.2, 0, .3, .4, 0, .2, .3], [.1, .2, 0, .3, .2, 0, .1]],
                         [[.3, .2, 0, .1, .3, .1, 0], [0, .3, .2, .1, 0, .3, .2]]],
                        dtype=dtype, device=device)[:, :batch].requires_grad_()
    times = (2**55 + 19 + torch.arange(7, device=device, dtype=torch.int64) * 4).expand(2, batch, 7)
    owners = torch.tensor([2, 0], device=device)
    control = torch.zeros_like(mass)
    control[:, :, 2::2] = (mass[:, :, 2::2] > 0).to(dtype)
    return (module, u, mass, times, owners, control), (u, mass, *module.parameters())


class QueryProjectionAudit(RuntimeAudit):
    """Observe only operators executed inside HeadQuery.forward."""
    def __init__(self):
        super().__init__()
        self.inside_query = False

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if self.inside_query:
            return super().__torch_dispatch__(func, types, args, kwargs)
        return func(*args, **(kwargs or {}))


def formula_state(checks, zero_mass_nonzeros):
    rows = list(checks.values())
    if any(row["nonfinite"] for row in rows):
        return "nonfinite"
    if any(zero_mass_nonzeros) or any(row["dtype_differences"] for row in rows):
        return "structural_failed"
    if any(row["metadata_failures"] for row in rows):
        return "metadata_failed"
    if any(row["support_differences"] for row in rows):
        return "support_failed"
    if any(row["threshold_failures"] for row in rows):
        return "finite_deviation"
    return "within_threshold"


def formula_witness(width=32, *, d=192, device="cpu", dtype=torch.float32, shared=True, batch=2):
    args, roots = fixture(width, d=d, device=device, dtype=dtype, shared=shared, batch=batch)
    audit = QueryProjectionAudit()
    query = args[0].query
    handles = [query.register_forward_pre_hook(lambda *unused: setattr(audit, "inside_query", True)),
               query.register_forward_hook(lambda *unused: setattr(audit, "inside_query", False))]
    try:
        with audit:
            actual = projected(*args)
    finally:
        for handle in handles:
            handle.remove()
    expected = direct_formula(*args)
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
    query_dispatch = audit.report()
    selected_fused = query.uses_fused_projection(args[1])
    observed_fused = width // 32 >= 4 and any(
        row["operator"] in ("aten.mm.default", "aten.bmm.default") and len(row["inputs"]) == 2
        and row["inputs"][0][-1] == d and row["inputs"][1][-2:] == [d, width]
        for row in query_dispatch["matrix_shapes"])
    state = formula_state(checks, zero)
    if selected_fused and not observed_fused and state != "nonfinite":
        state = "structural_failed"
    return dict(width=width, d=d, summary_width=d//4, ffn_width=4*d, heads=width // 32,
                head_width=32, batch=batch, input_shape=list(args[1].shape), sharing="shared_base_private_rank8" if shared else "independent",
                state=state,
                scope="FP32 or bounded CPU FP64 literal formula, private adapters and node state, int64 translation, zero write, window eviction, full VJP",
                query_dispatch=dict(selected_fused_projection=selected_fused,
                                    fused_base_projection_observed=observed_fused, **query_dispatch),
                identity=dict(parameters=fingerprint(args[0].state_dict()), input=fingerprint(args[1]),
                              mass=fingerprint(args[2]), times=fingerprint(args[3])),
                zero_mass_gradient_nonzeros=zero, first_time=int(args[3][0, 0, 0]),
                retained_times=[[list(bank) for bank in row] for row in actual["state"]], checks=checks)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--d", type=int, default=192)
    parser.add_argument("--widths", type=int, nargs="+", default=[96, 64, 32])
    parser.add_argument("--batch", type=int, choices=(1, 2), default=1)
    parser.add_argument("--shared", action="store_true", help="also use shared bases with private rank8 adapters; default independent")
    parser.add_argument("--precision", choices=("fp32", "fp64"), default="fp32")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.threads < 1 or len(args.widths) != len(set(args.widths)):
        parser.error("choose a new output, positive thread count and distinct widths")
    if args.d < 32 or args.d % 32 or any(width <= 0 or width > args.d or width % 32 for width in args.widths):
        parser.error("d and widths must be positive multiples of 32 with widths <= d")
    if args.precision == "fp64" and args.device != "cpu":
        parser.error("FP64 is a bounded CPU diagnostic")
    if args.device.startswith("npu"):
        import torch_npu
        torch.npu.set_device(args.device)
        torch.npu.matmul.allow_hf32 = False
        torch.npu.conv.allow_hf32 = False
    torch.set_num_threads(args.threads)
    dtype = torch.float64 if args.precision == "fp64" else torch.float32
    report = dict(schema="tide-memory-width-formulas-v1", state="running",
                  source=os.environ.get("TIDE_SOURCE_COMMIT") or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  physical_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"), torch=str(torch.__version__),
                  cpu_affinity=sorted(os.sched_getaffinity(0)),
                  precision=precision_policy(args.device, "fp32") if args.precision == "fp32" else dict(requested="fp64", purpose="bounded CPU diagnostic"),
                  thresholds=THRESHOLDS[args.precision], rows=[])
    atomic(args.output, report)
    for width in args.widths:
        try:
            row = formula_witness(width, d=args.d, device=args.device, dtype=dtype, shared=args.shared, batch=args.batch)
        except Exception as error:
            import traceback
            traceback.print_exc()
            row = dict(width=width, state="execution_failed", error=repr(error))
        report["rows"].append(row)
        atomic(args.output, report)
    order = ("execution_failed", "nonfinite", "structural_failed", "metadata_failed", "support_failed", "finite_deviation")
    report["state"] = next((state for state in order if any(row["state"] == state for row in report["rows"])), "within_threshold")
    atomic(args.output, report)
    print(json.dumps(dict(state=report["state"], widths=[dict(width=row["width"], state=row["state"]) for row in report["rows"]])))
    return int(report["state"] not in ("within_threshold", "finite_deviation"))


if __name__ == "__main__":
    raise SystemExit(main())
