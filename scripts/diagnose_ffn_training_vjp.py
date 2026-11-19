"""One real CPU training VJP, then a bounded same-input K-down diagnosis.

Use a frozen implementation and an existing execution-qualification directory.
The original inference warmup and signed-cosine objective are reused, followed
by one baseline backward. Only its last independent K FFN down call is retained.
The saved W-path observation is a historical comparator, not a second new
whole-model run. Local gather/grouped replays share exactly X, W and incoming dY.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import traceback

import torch

from tide.parameters import NodeLinear, NodeOwners, owner_ids


MODULE = "memories.K.ffn.down"
MAX_CAPTURE_BYTES = 128 << 20


def tensor_identity(value):
    value = value.detach().cpu().contiguous()
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())


def compare_tensor(actual, expected, tolerance):
    """Evaluate errors in FP64 without changing the original tolerance."""
    if actual.shape != expected.shape:
        return dict(state="shape_mismatch", actual=list(actual.shape), expected=list(expected.shape))
    a, b = actual.detach().double(), expected.detach().double()
    valid = torch.isfinite(a) & torch.isfinite(b)
    nonfinite = int((~valid).sum())
    finite = nonfinite == 0
    delta = (a - b).abs()
    scale = delta / (tolerance["atol"] + tolerance["rtol"] * b.abs())
    index = (int((~valid).reshape(-1).nonzero()[0]) if nonfinite else
             int(scale.reshape(-1).argmax()) if scale.numel() else 0)
    failed = nonfinite + int(((scale > 1) & valid).sum())
    def number(value):
        value = float(value)
        return value if math.isfinite(value) else None
    return dict(state="nonfinite" if nonfinite else "within_threshold" if not failed else "outside_threshold",
                exact=torch.equal(a, b), finite=finite, **tolerance,
                elements=a.numel(), failed_elements=failed, nonfinite_pairs=nonfinite,
                nonfinite_actual=int((~torch.isfinite(a)).sum()), nonfinite_expected=int((~torch.isfinite(b)).sum()),
                max_abs=number(delta[valid].max()) if bool(valid.any()) else None,
                max_scaled=number(scale[valid].max()) if bool(valid.any()) else None,
                worst_flat_index=index,
                worst_actual=number(a.reshape(-1)[index]) if a.numel() else None,
                worst_expected=number(b.reshape(-1)[index]) if b.numel() else None,
                maxima_scope="finite element pairs only; nonfinite pairs are separate failures",
                arithmetic="CPU FP64 difference/scaling of retained values; no tolerance widening")


def literal_linear_vjp(x, weight, ids, incoming):
    """Independent explicit FP64 y=XW^T, dx=dYW, dW=sum(dY^T X).

    Summation is per invocation row, including repeated owners; unselected
    owners receive exactly zero. This function uses no production projection
    or autograd implementation.
    """
    x, weight, incoming = x.double(), weight.double(), incoming.double()
    flat_x = x.reshape(len(ids), -1, weight.shape[-1])
    flat_dy = incoming.reshape(len(ids), -1, weight.shape[-2])
    ys, dxs = [], []
    dw = torch.zeros_like(weight)
    for index, owner in enumerate(ids.tolist()):
        ys.append(flat_x[index] @ weight[owner].T)
        dxs.append(flat_dy[index] @ weight[owner])
        dw[owner] += flat_dy[index].T @ flat_x[index]
    return dict(output=torch.stack(ys).reshape(*x.shape[:-1], weight.shape[-2]),
                input_vjp=torch.stack(dxs).reshape_as(x), weight_vjp=dw)


class LastDownCall:
    """Tensor hooks observe the existing graph without wrapping module outputs."""
    def __init__(self, layer, max_bytes=MAX_CAPTURE_BYTES):
        if not isinstance(layer, NodeLinear) or layer.shared or layer.bias is not None:
            raise ValueError("this diagnostic is restricted to an independent bias-free NodeLinear")
        if not 0 < max_bytes <= MAX_CAPTURE_BYTES:
            raise ValueError("capture budget must be positive and at most 128 MiB")
        self.layer, self.max_bytes = layer, max_bytes
        self.calls, self.last, self.handles = 0, None, []

    def __enter__(self):
        self.forward_handle = self.layer.register_forward_hook(self._capture)
        return self

    def _capture(self, module, arguments, output):
        x, owners = arguments
        if not torch.is_grad_enabled():
            return
        self.calls += 1
        index = self.calls
        ids = owner_ids(owners)
        possible = owners.possible_ids if isinstance(owners, NodeOwners) else tuple(range(module.nodes))
        exact = owners.exact_ids if isinstance(owners, NodeOwners) else None
        self.last = dict(index=index, x=x, output=output, ids=ids, possible=possible, exact=exact)
        def observe_incoming(value):
            if self.last["index"] == index:
                self._check_before_clone()
                self.last["incoming_layout"] = self._layout(value)
                self.last["incoming"] = value.detach().clone()
        def observe_input(value):
            if self.last["index"] == index:
                self._check_before_clone()
                self.last["input_vjp_layout"] = self._layout(value)
                self.last["input_vjp"] = value.detach().clone()
        self.handles.extend([output.register_hook(observe_incoming), x.register_hook(observe_input)])

    @staticmethod
    def _layout(value):
        return dict(shape=list(value.shape), stride=list(value.stride()),
                    storage_offset=value.storage_offset(), contiguous=value.is_contiguous(), dtype=str(value.dtype))

    def _check_before_clone(self):
        row = self.last
        start, stop = min(row["possible"]), max(row["possible"]) + 1
        weight = self.layer.weight[start:stop]
        # Incoming/input cotangents have their differentiated tensor's shape
        # and dtype. Charge both, X/Y and the bank before either hook clones.
        total = sum(value.numel() * value.element_size() * copies for value, copies in
                    ((row["x"], 2), (row["output"], 2), (weight, 1), (row["ids"], 1)))
        if total + (1 << 20) > self.max_bytes:
            raise MemoryError(f"last down-call witness needs {total} tensor bytes; capture budget {self.max_bytes}; rejected before cotangent clone")

    def finish(self):
        if self.last is None or not {"incoming", "input_vjp"} <= self.last.keys():
            raise RuntimeError("the final down call did not receive both actual backward cotangents")
        row = self.last
        start, stop = min(row["possible"]), max(row["possible"]) + 1
        weight = self.layer.weight[start:stop]
        values = dict(x=row["x"], output=row["output"], incoming=row["incoming"],
                      input_vjp=row["input_vjp"], weight=weight, ids=row["ids"] - start)
        total = sum(value.numel() * value.element_size() for value in values.values())
        # Reserve one MiB for serialization metadata, rather than silently
        # truncating the selected call or retaining an earlier/smaller call.
        if total + (1 << 20) > self.max_bytes:
            raise MemoryError(f"last down-call witness needs {total} tensor bytes; capture budget {self.max_bytes}")
        original_layout = {key: self._layout(value) for key, value in values.items()}
        original_layout.update(incoming=row["incoming_layout"], input_vjp=row["input_vjp_layout"])
        values = {key: value.detach().cpu().clone() for key, value in values.items()}
        return dict(**values, possible_ids=tuple(owner - start for owner in row["possible"]),
                    exact_ids=None if row["exact"] is None else tuple(owner - start for owner in row["exact"]),
                    original_bank_start=start, original_bank_stop=stop,
                    original_owner_ids=row["ids"].detach().cpu().tolist(),
                    call_index=row["index"], observed_forward_calls=self.calls, tensor_bytes=total,
                    original_layout=original_layout,
                    layout="Only the statically bounded bank slice is copied; owner IDs are translated by its start. Repeated and absent owners remain unchanged.")

    def __exit__(self, *exc):
        self.forward_handle.remove()
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.last = None


def local_replay(capture, access):
    layer = NodeLinear(capture["weight"].shape[0], capture["weight"].shape[-1],
                       capture["weight"].shape[-2], weight_access=access).to(capture["x"].dtype)
    with torch.no_grad():
        layer.weight.copy_(capture["weight"])
    x = capture["x"].clone().requires_grad_()
    owners = NodeOwners(capture["ids"], capture["possible_ids"], capture["exact_ids"])
    output = layer(x, owners)
    dx, dw = torch.autograd.grad(output, (x, layer.weight), capture["incoming"])
    return dict(output=output.detach(), input_vjp=dx.detach(), weight_vjp=dw.detach())


def backbone_scope(observation):
    return dict(loss=observation["loss"], input_gradient=observation["input_gradient"],
                segments=[{key: segment[key] for key in ("values", "strengths", "times")}
                          for segment in observation["forward"]["segments"]])


def artifact_identity(directory, original, relative):
    path = directory / relative
    identity = next(row for row in original["artifacts"] if row["path"] == relative)
    info = path.stat()
    if not identity.get("retained") or info.st_size != identity["bytes"]:
        raise ValueError("original retained artifact is missing or has a different size: " + relative)
    return dict(recorded=identity, current_stat=dict(device=info.st_dev, inode=info.st_ino,
                bytes=info.st_size, mtime_ns=info.st_mtime_ns),
                hash_policy="Original qualification SHA256 retained; this diagnostic does not rehash the whole multi-GiB input artifact.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification", type=Path, required=True, help="original qualification/result.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, choices=range(1, 33), default=32)
    parser.add_argument("--capture-mib", type=int, choices=range(2, 129), default=128)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".tensors.pt").exists():
        parser.error("choose a new output identity")
    from benchmarks.inference import source_identity
    from benchmarks.workloads import tensor_hash
    from scripts.check_execution_candidates import restore_extra
    from scripts.check_training_paths import build_candidate, compare, observe
    from scripts.job import atomic
    from scripts.qualify_precision import atomic_tensor
    from tide import ModelConfig
    original_bytes = args.qualification.read_bytes()
    original = json.loads(original_bytes)
    if original["device"] != "cpu" or original["precision"] != "fp32" or original["parameters"] > 200_000_000:
        raise ValueError("only an existing bounded CPU FP32 execution qualification is supported")
    if args.threads != original["threads"] or sorted(os.sched_getaffinity(0)) != original["cpu_affinity"]:
        raise ValueError("restore original thread count and CPU affinity before replay")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    source = source_identity()
    if source["tide_content_sha256"] != original["source"]["tide_content_sha256"]:
        raise ValueError("production source differs from the original qualification")
    directory = args.qualification.parent
    cases = {case["name"]: case for case in original["candidates"]}
    record = dict(schema="tide-ffn-training-vjp-diagnosis-v1", state="running", source=source,
        original_result=dict(path=str(args.qualification), sha256=hashlib.sha256(original_bytes).hexdigest(),
                             state=original["state"], error=original.get("error")),
        inputs={name: artifact_identity(directory, original, name) for name in
                ("anchor.pt", "baseline/vjp1.pt", "weight_access/vjp1.pt")},
        scope="One baseline real backward; final independent memories.K.ffn.down; same captured X/W/dY gather/grouped and independent FP64 adjoints. No optimizer update or new whole-model W path.",
        tolerance=original["tolerance"], cpu_affinity=sorted(os.sched_getaffinity(0)),
        threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(), torch=str(torch.__version__),
        original_objective=original["objective"], capture_limit_bytes=args.capture_mib << 20,
        attribution_boundary="A mismatch against original saved outputs/selected gradients is retained and prevents assuming historical reproducibility or a rounding cause.")
    atomic(args.output, record)
    try:
        payload = torch.load(directory / "anchor.pt", map_location="cpu", weights_only=False, mmap=True)
        if payload["config"] != original["base_config"] or tensor_hash(payload["inputs"][1]) != original["input_hashes"][1]:
            raise ValueError("anchor configuration or actual input differs from recorded identity")
        base = ModelConfig.from_dict(payload["config"])
        model, optimizer, state = build_candidate(base, "baseline", payload["training"], [torch.device("cpu")],
                                                   "fp32", "auto", overrides=original["resolved_candidates"]["baseline"])
        if model.config.to_dict() != cases["baseline"]["config"]:
            raise ValueError("resolved baseline configuration differs from original")
        if model.config.checkpoint_regions:
            raise ValueError("this final-forward-call witness does not support FFN re-execution inside region checkpoint backward")
        restore_extra(model, payload["extra"], torch.device("cpu"))
        # Reproduce the inference phase immediately preceding the original vjp1.
        warm, _, _ = observe(model, optimizer, state, payload["inputs"][1], "fp32", [torch.device("cpu")],
                              original["workload"]["chunk"], grad=False)
        del warm
        gc.collect()
        layer = dict(model.named_modules())[MODULE]
        with LastDownCall(layer, args.capture_mib << 20) as recorder:
            observed, _, work = observe(model, optimizer, state, payload["inputs"][1], "fp32", [torch.device("cpu")],
                                        original["workload"]["chunk"])
            captured = recorder.finish()
        selected = torch.tensor(sorted(set(captured["original_owner_ids"])), dtype=torch.int64)
        actual_gradient = observed["gradients"][MODULE + ".weight"].index_select(0, selected)
        record["capture"] = {key: value for key, value in captured.items() if not isinstance(value, torch.Tensor)}
        record["capture"]["tensor_identities"] = {key: tensor_identity(value) for key, value in captured.items() if isinstance(value, torch.Tensor)}
        record["actual_work"] = work
        record["original_vjp1_checks"] = {}
        historical_gradients = {}
        for name in ("baseline", "weight_access"):
            prior = torch.load(directory / name / "vjp1.pt", map_location="cpu", weights_only=False, mmap=True)
            prior_gradient = prior["gradients"][MODULE + ".weight"].index_select(0, selected)
            historical_gradients[name] = prior_gradient
            row = dict(original_saved_observation=cases[name]["observations"]["vjp1"]["summary"],
                       selected_owner_rows=selected.tolist(), saved_selected_gradient=tensor_identity(prior_gradient))
            if name == "baseline":
                row.update(backbone_and_input_vjp=compare(backbone_scope(observed), backbone_scope(prior), original["tolerance"]),
                           selected_parameter_gradient=compare_tensor(actual_gradient, prior_gradient, original["tolerance"]),
                           actual_selected_gradient=tensor_identity(actual_gradient),
                           whole_artifact_rehashed=False)
            else:
                row["scope"] = "Saved W-path gradient is a historical comparator only; no new W whole-model forward/backward was run. Local replay uses baseline X/W/dY."
                row["saved_W_vs_replayed_baseline_selected_gradient"] = compare_tensor(prior_gradient, actual_gradient, original["tolerance"])
            record["original_vjp1_checks"][name] = row
            del prior
        record["historical_selected_gradient_W_vs_baseline"] = compare_tensor(historical_gradients["weight_access"], historical_gradients["baseline"], original["tolerance"])
        del observed, model, optimizer, state, payload, historical_gradients
        gc.collect()
        reference = literal_linear_vjp(captured["x"], captured["weight"], captured["ids"], captured["incoming"])
        replays = {access: local_replay(captured, access) for access in ("gather", "grouped")}
        record["local_against_fp64_formula"] = {access: {key: compare_tensor(value, reference[key], original["tolerance"])
            for key, value in replay.items()} for access, replay in replays.items()}
        record["local_grouped_vs_gather"] = {key: compare_tensor(replays["grouped"][key], replays["gather"][key], original["tolerance"])
                                               for key in reference}
        record["actual_call_against_replay"] = {access: {key: compare_tensor(captured[key], replay[key], original["tolerance"])
            for key in ("output", "input_vjp")} for access, replay in replays.items()}
        record["local_parameter_scope"] = "weight_vjp is this single call's contribution, summed over repeated owners. Original saved whole-model gradients include other regions/chunks and are checked separately on selected owner rows."
        tensor_path = args.output.with_suffix(".tensors.pt")
        atomic_tensor(tensor_path, captured)
        if tensor_path.stat().st_size > args.capture_mib << 20:
            raise MemoryError("serialized single-call witness exceeded its explicit capture budget")
        record["retained_call"] = dict(path=str(tensor_path), bytes=tensor_path.stat().st_size,
                                       sha256=hashlib.sha256(tensor_path.read_bytes()).hexdigest())
        checks = record["original_vjp1_checks"]["baseline"]
        exact = checks["backbone_and_input_vjp"]["comparison"]["exact"] and checks["selected_parameter_gradient"]["exact"]
        record.update(state="completed_bounded_diagnosis", original_selected_scope_exact=exact,
                      replay_limit="Local replay changes no production source. FP64 is a diagnostic root for this retained call, not a training precision or whole-model equivalence proof.")
        atomic(args.output, record)
        print(json.dumps(dict(state=record["state"], original_selected_scope_exact=exact, output=str(args.output))))
    except Exception as error:
        record.update(state="failed_bounded_diagnosis", error=repr(error), traceback=traceback.format_exc())
        atomic(args.output, record)
        raise


if __name__ == "__main__":
    main()
