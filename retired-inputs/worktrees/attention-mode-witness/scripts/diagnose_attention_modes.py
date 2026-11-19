"""Bounded first-Attention witness at an existing FP32 training anchor.

Two serial forwards compare numeric call fingerprints; at most one prefix replay
recovers the first changed call. Only that call is retained, followed by two
same-input primitive calls and a few independent scalar FP64 query rows. This is
intrusive forward diagnosis, not VJP qualification or performance measurement.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
from unittest.mock import patch

import torch
import tide
from torch.nn import functional as F

from tide import ModelConfig
from scripts.check_training_paths import (THRESHOLDS, arithmetic, build_candidate,
    configure_devices, forward, package_identity, screen, training_data)
from scripts.job import atomic
from scripts.qualify_precision import atomic_tensor, file_sha256


NAMES = ("q", "k", "v", "weights", "mask")


def describe(value):
    if value is None:
        return None
    value = value.detach().cpu().contiguous()
    return dict(shape=list(value.shape), dtype=str(value.dtype),
        sha256=hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())


def numeric(actual, expected):
    """Original FP32 tolerance, evaluated against a CPU FP64 reference."""
    if actual.shape != expected.shape:
        return dict(shape_matches=False)
    a, b = actual.detach().cpu().double(), expected.detach().cpu().double()
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        return dict(shape_matches=True, finite=False)
    delta = (a-b).abs()
    scaled = delta/(THRESHOLDS["fp32"]["atol"] + THRESHOLDS["fp32"]["rtol"]*b.abs())
    return dict(shape_matches=True, finite=True, exact=torch.equal(a, b),
        max_abs=float(delta.max()) if delta.numel() else 0.,
        max_scaled=float(scaled.max()) if scaled.numel() else 0.,
        outside_threshold=int((scaled > 1).sum()),
        relative_l2=float(delta.norm()/b.norm().clamp_min(1e-30)))


@contextmanager
def dispatch(device):
    """Observe actual adapter/custom-forward/native API calls without choosing one."""
    module = importlib.import_module("tide.attention")
    native, tiled, sdpa = module._native_attention, module._TiledAttention.forward, F.scaled_dot_product_attention
    result = dict(effective="math", representation="explicit_weight_quotient")
    def native_call(q, k, v, weights, mask, backend, **kwargs):
        result.update(effective=backend, adapter_width=q.shape[-1],
            adapter_prior_requires_grad=weights.requires_grad,
            omitted_unit_prior=kwargs.get("unit_prior", False))
        return native(q, k, v, weights, mask, backend, **kwargs)
    def tiled_call(*args, **kwargs):
        result.update(effective="tiled", representation="explicit_weight_quotient")
        return tiled(*args, **kwargs)
    def sdpa_call(q, k, v, *args, **kwargs):
        bias = kwargs.get("attn_mask", args[0] if args else None)
        result.update(native_api="torch.nn.functional.scaled_dot_product_attention",
            native_qkv_shapes=[list(x.shape) for x in (q, k, v)],
            native_qkv_dtypes=[str(x.dtype) for x in (q, k, v)],
            native_mask_dtype=None if bias is None else str(bias.dtype),
            representation="unit_or_boolean_mask" if bias is None or bias.dtype == torch.bool else "additive_log_prior_bias")
        return sdpa(q, k, v, *args, **kwargs)
    with ExitStack() as stack:
        stack.enter_context(patch.object(module, "_native_attention", native_call))
        stack.enter_context(patch.object(module._TiledAttention, "forward", staticmethod(tiled_call)))
        stack.enter_context(patch.object(F, "scaled_dot_product_attention", sdpa_call))
        if torch.device(device).type == "npu":
            vendor = importlib.import_module("torch_npu")
            fusion = vendor.npu_fusion_attention_v2
            def fusion_call(q, k, v, *args, **kwargs):
                pse = kwargs.get("pse")
                result.update(native_api="torch_npu.npu_fusion_attention_v2",
                    native_qkv_shapes=[list(x.shape) for x in (q, k, v)],
                    native_qkv_dtypes=[str(x.dtype) for x in (q, k, v)],
                    native_pse=None if pse is None else dict(shape=list(pse.shape), dtype=str(pse.dtype)),
                    native_pse_type=kwargs.get("pse_type"), native_scale=kwargs.get("scale"),
                    representation="additive_log_prior_pse" if pse is not None else
                        "rank_one_qk_extension" if q.shape[-1] > result["adapter_width"] else "unit_prior")
                return fusion(q, k, v, *args, **kwargs)
            stack.enter_context(patch.object(vendor, "npu_fusion_attention_v2", fusion_call))
        yield result


class StopCapture(Exception):
    pass


class CaptureLimit(Exception):
    pass


class Calls:
    """Hash-only bounded index in memory; persisted evidence is one call only."""
    def __init__(self, device, *, elements, max_calls, reference=None, target=None):
        self.device, self.elements, self.max_calls = device, elements, max_calls
        self.reference, self.target = reference, target
        self.phase, self.rows, self.selected = "", [], None
        self.prefix = hashlib.sha256()
        self.path_changes = 0

    @contextmanager
    def install(self):
        module = importlib.import_module("tide.attention")
        self.original = module.weighted_attention
        with ExitStack() as stack:
            for site in ("attention", "reception", "kv", "region", "region_batch"):
                owner = importlib.import_module("tide."+site)
                def wrapped(q, k, v, weights=None, mask=None, backend="auto", *, _site=site):
                    return self.call(_site, (q, k, v, weights, mask), backend)
                stack.enter_context(patch.object(owner, "weighted_attention", wrapped))
            yield self

    def call(self, site, values, backend):
        index = len(self.rows)
        if index >= self.max_calls:
            raise CaptureLimit("Attention call count exceeds "+str(self.max_calls))
        q, k, v, weights, mask = values
        prefix = torch.broadcast_shapes(q.shape[:-2], k.shape[:-2], v.shape[:-2])
        output_elements = math.prod(prefix)*q.shape[-2]*v.shape[-1]
        elements = sum(x.numel() for x in values if x is not None) + output_elements
        if elements > self.elements:
            raise CaptureLimit(f"call {index} needs {elements} input/output elements; cap is {self.elements}")
        inputs = {name: describe(value) for name, value in zip(NAMES, values)}
        flags = {name: value is not None and value.requires_grad for name, value in zip(NAMES, values)}
        with dispatch(self.device) as path:
            output = self.original(*values, backend=backend)
        if q.shape[-2] == 0 or k.shape[-2] == 0:
            path.update(effective="empty", representation="empty_output")
        row = dict(index=index, phase=self.phase, site=site, inputs=inputs, output=describe(output),
                   requested_backend=backend, requires_grad=flags, grad_enabled=torch.is_grad_enabled(),
                   output_requires_grad=output.requires_grad, dispatch=path)
        identity = {key: row[key] for key in ("index", "phase", "site", "inputs", "output")}
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        row["numeric_sha256"] = digest
        expected = None if self.reference is None or index >= len(self.reference) else self.reference[index]
        changed = self.reference is not None and (expected is None or expected["numeric_sha256"] != digest)
        dispatch_keys = ("effective", "representation", "native_qkv_shapes", "native_pse_type")
        if expected is not None and any(expected["dispatch"].get(key) != path.get(key) for key in dispatch_keys):
            self.path_changes += 1
        self.rows.append(row)
        if changed or index == self.target:
            self.selected = dict(metadata=row, inputs={name: None if value is None else value.detach().cpu().clone()
                for name, value in zip(NAMES, values)}, output=output.detach().cpu().clone())
            raise StopCapture()
        self.prefix.update(digest.encode())
        return output


def expand_auxiliary(value, shape, *, prior):
    """Index views for the documented compact-key-prior/broadcast-mask interface."""
    if value is None:
        return torch.ones((), dtype=torch.float64 if prior else torch.bool).expand(shape)
    choices = [value]
    if prior and value.ndim and value.ndim <= len(shape)-2:
        choices.insert(0, value.reshape(*value.shape[:-1], 1, 1, value.shape[-1]))
    if value.ndim == len(shape)-1:
        choices.append(value.unsqueeze(-3))
    for candidate in choices:
        try:
            return torch.broadcast_to(candidate, shape)
        except RuntimeError:
            continue
    raise ValueError("oracle auxiliary layout is not supported")


def scalar_rows(inputs, rows):
    """Independent stable weighted exponential quotient, scalar IEEE FP64 only.

    Enumerates every visible key and every output coordinate in selected complete
    query rows. No production Attention, tensor matmul, softmax or autograd.
    """
    q, k, v = (inputs[name].double() for name in ("q", "k", "v"))
    prefix = torch.broadcast_shapes(q.shape[:-2], k.shape[:-2], v.shape[:-2])
    q, k, v = [torch.broadcast_to(x, prefix+x.shape[-2:]) for x in (q, k, v)]
    shape = prefix+(q.shape[-2], k.shape[-2])
    weights = expand_auxiliary(inputs["weights"], shape, prior=True)
    mask = expand_auxiliary(inputs["mask"], shape, prior=False)
    answer = []
    for flat in rows:
        index, remaining = [], flat
        for size in reversed(shape[:-1]):
            index.insert(0, remaining % size); remaining //= size
        if remaining:
            raise ValueError("oracle query row is out of range")
        index = tuple(index)
        query, keys, values = q[index].tolist(), k[index[:-1]].tolist(), v[index[:-1]].tolist()
        prior, visible = weights[index].tolist(), mask[index].tolist()
        scores = [math.fsum(a*b for a, b in zip(query, key))/math.sqrt(q.shape[-1]) + math.log(w)
                  if active and w > 0 else -math.inf for key, w, active in zip(keys, prior, visible)]
        maximum = max(scores, default=-math.inf)
        terms = [math.exp(score-maximum) if score != -math.inf else 0. for score in scores]
        denominator = math.fsum(terms)
        answer.append([math.fsum(term*value[channel] for term, value in zip(terms, values))/denominator
                       if denominator else 0. for channel in range(v.shape[-1])])
    return torch.tensor(answer, dtype=torch.float64)


def replay_pair(reference, changed, device, row_limit):
    from tide.attention import weighted_attention
    outputs, reports = {}, {}
    for name, captured in (("inference", reference), ("grad_forward", changed)):
        metadata = captured["metadata"]
        inputs = {key: None if value is None else value.to(device).clone().requires_grad_(metadata["requires_grad"][key])
                  for key, value in reference["inputs"].items()}
        with torch.set_grad_enabled(metadata["grad_enabled"]), arithmetic(device, "fp32"), dispatch(device) as path:
            output = weighted_attention(*(inputs[key] for key in NAMES), backend=metadata["requested_backend"])
        outputs[name] = output.detach().cpu()
        same_inputs = metadata["inputs"] == reference["metadata"]["inputs"]
        reports[name] = dict(dispatch=path, same_inputs_as_capture=same_inputs,
            matches_capture=numeric(outputs[name], captured["output"]) if same_inputs else None,
            output=describe(outputs[name]), output_requires_grad=output.requires_grad)
    a, b = (outputs[name].reshape(-1, outputs[name].shape[-1]) for name in ("inference", "grad_forward"))
    if not a.shape[0]:
        return dict(state="empty_call", replays=reports), dict(outputs=outputs)
    error = ((a.double()-b.double()).abs()/(1e-6+.002*b.double().abs())).amax(-1)
    worst = int(error.argmax())
    rows = [worst] + [index for index in range(min(row_limit, a.shape[0])) if index != worst]
    rows = rows[:row_limit]
    oracle = scalar_rows(reference["inputs"], rows)
    return dict(state="completed", primitive_replays=2,
        common_input="captured inference inputs, preserving each mode's requires_grad flags",
        replays=reports, mode_difference=numeric(a, b), oracle=dict(
            method="independent scalar FP64 dot/weighted exponential quotient using math.fsum; all keys and output channels per selected row",
            row_indices=rows, inference=numeric(a[rows], oracle), grad_forward=numeric(b[rows], oracle))), dict(outputs=outputs, oracle_rows=rows, oracle=oracle)


def diagnose(bundle, *, device, chunk, elements=2_000_000, max_calls=4096, oracle_rows=8):
    if not (1 <= elements <= 20_000_000 and 1 <= max_calls <= 8192 and 1 <= oracle_rows <= 32):
        raise ValueError("bounded capture requires elements1..20000000, calls1..8192, oracle rows1..32")
    config = ModelConfig.from_dict(bundle["config"])
    anchor, value = bundle["anchor"], bundle["inputs"][1]
    if anchor["state"] is None or not anchor["optimizer"]["state"]:
        raise ValueError("a real nonempty training anchor is required")
    model, optimizer, state = build_candidate(config, "baseline", anchor, [device], "fp32", "auto")
    frozen = screen(training_data(model, optimizer, state))["sha256"]
    if frozen != screen(anchor)["sha256"]:
        raise RuntimeError("loaded anchor differs")
    scans = []
    def execute(calls, grad):
        scans.append(calls)
        try:
            with torch.set_grad_enabled(grad), arithmetic(device, "fp32"), calls.install():
                result, loss = forward(model, value.to(device).clone().requires_grad_(grad), state, chunk, trace=calls)
                del result, loss
        except StopCapture:
            pass
        if screen(training_data(model, optimizer, state))["sha256"] != frozen:
            raise RuntimeError("forward changed the fixed parameter/state/optimizer anchor")
    baseline = Calls(device, elements=elements, max_calls=max_calls)
    report, payload = dict(state="running", limits=dict(elements_per_call=elements, calls_per_forward=max_calls,
        oracle_rows=oracle_rows, model_forwards=3, primitive_replays=2)), {}
    try:
        execute(baseline, False)
        candidate = Calls(device, elements=elements, max_calls=max_calls, reference=baseline.rows)
        execute(candidate, True)
        report.update(reference_calls=len(baseline.rows), compared_calls=len(candidate.rows),
            equal_prefix_sha256=candidate.prefix.hexdigest(), dispatch_changes=candidate.path_changes)
        if candidate.selected is None:
            report["state"] = "no_attention_difference" if len(candidate.rows) == len(baseline.rows) else "call_count_changed"
            return report, payload
        index = candidate.selected["metadata"]["index"]
        if index >= len(baseline.rows):
            report.update(state="unpaired_call", first=candidate.selected["metadata"])
            return report, dict(grad_forward=candidate.selected)
        recovery = Calls(device, elements=elements, max_calls=max_calls, reference=baseline.rows, target=index)
        execute(recovery, False)
        if recovery.selected is None or recovery.selected["metadata"]["numeric_sha256"] != baseline.rows[index]["numeric_sha256"]:
            report.update(state="inference_repeat_changed", reason="single bounded recovery did not reproduce the selected reference call")
            return report, dict(grad_forward=candidate.selected, repeated_inference=recovery.selected)
        reference, changed = recovery.selected, candidate.selected
        report.update(state="first_attention_difference", first_index=index, equal_earlier_calls=index,
            inference=reference["metadata"], grad_forward=changed["metadata"],
            input_values_equal=reference["metadata"]["inputs"] == changed["metadata"]["inputs"],
            captured_output_difference=numeric(reference["output"], changed["output"]), inference_repeat_exact=True)
        payload.update(inference=reference, grad_forward=changed)
        report["same_input"], payload["same_input"] = replay_pair(reference, changed, device, oracle_rows)
        return report, payload
    except CaptureLimit as error:
        report.update(state="capture_limit", reason=str(error), calls_per_attempt=[len(scan.rows) for scan in scans])
        return report, payload
    finally:
        report["actual_model_forward_attempts"] = len(scans)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--source-record", type=Path, help="defaults to result.json beside anchor.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--capture-elements", type=int, default=2_000_000)
    parser.add_argument("--max-calls", type=int, default=4096)
    parser.add_argument("--oracle-rows", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("choose a new output directory")
    if not (1 <= args.capture_elements <= 20_000_000 and 1 <= args.max_calls <= 8192 and 1 <= args.oracle_rows <= 32):
        parser.error("bounded capture requires elements1..20000000, calls1..8192, oracle rows1..32")
    if Path(tide.__file__).resolve().parent != Path(__file__).resolve().parents[1]/"src"/"tide":
        parser.error("prepend the frozen checkout src to PYTHONPATH")
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    source_path = args.source_record or args.anchor.with_name("result.json")
    original = json.loads(source_path.read_text())
    if original["arguments"]["precision"] != "fp32":
        parser.error("this targeted mode diagnostic requires the original FP32 anchor")
    if original["package_identity"]["sha256"] != package_identity()["sha256"]:
        parser.error("production package differs from the recorded anchor implementation")
    bundle = torch.load(args.anchor, map_location="cpu", weights_only=False)
    if screen(bundle["anchor"])["sha256"] != original["result"]["fixed_start"]["sha256"]:
        parser.error("anchor content does not match the source record")
    from benchmarks.workloads import tensor_hash
    if tensor_hash(bundle["inputs"][1]) != original["result"]["input_hashes"][1]:
        parser.error("inputs[1] does not match the source record")
    baseline = original["result"]["candidates"][0]
    if baseline["name"] != "baseline" or bundle["config"] != baseline["config"]:
        parser.error("anchor configuration does not match the recorded baseline")
    value = bundle["inputs"][1]
    if not (value.ndim == 3 and 1 <= value.shape[0] <= 8 and 1 <= value.shape[1] <= 64
            and 1 <= original["arguments"]["chunk"] <= value.shape[1] and value.dtype == torch.float32):
        parser.error("expected the original bounded FP32 B1..8/L1..64 input and valid chunk")
    parameters = sum(value.numel() for value in bundle["anchor"]["parameters"].values())
    if parameters > 20_000_000:
        parser.error("anchor exceeds the fixed 20 million parameter/buffer element cap")
    configure_devices([args.device])
    deterministic = original.get("deterministic_algorithms", False)
    torch.use_deterministic_algorithms(deterministic)
    args.output_dir.mkdir(parents=True)
    record = dict(source=os.environ.get("TIDE_SOURCE_COMMIT") or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        harness_sha256=file_sha256(__file__), package_identity=package_identity(),
        source_record=str(source_path.resolve()), source_record_sha256=file_sha256(source_path),
        anchor=str(args.anchor.resolve()), anchor_sha256=file_sha256(args.anchor), original_source=original["source"],
        original_mode_comparison=baseline["checks"]["inference_vs_grad_forward"]["comparison"],
        input_index=1, input=describe(bundle["inputs"][1]), config=bundle["config"],
        device=args.device, torch=str(torch.__version__), cpu_affinity=sorted(os.sched_getaffinity(0)),
        physical_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"), resource_identity=os.environ.get("TIDE_RESOURCE_IDENTITY"),
        deterministic_algorithms=deterministic, tolerance=THRESHOLDS["fp32"], state="running",
        scope="first changed numeric Attention call in serial traversal; earlier non-Attention operators are not captured; no VJP or global causal claim")
    if torch.device(args.device).type == "npu":
        import torch_npu
        record.update(torch_npu=str(torch_npu.__version__),
            allow_hf32=dict(matmul=torch.npu.matmul.allow_hf32, conv=torch.npu.conv.allow_hf32))
    atomic(args.output_dir/"result.json", record)
    try:
        result, payload = diagnose(bundle, device=args.device, chunk=original["arguments"]["chunk"],
            elements=args.capture_elements, max_calls=args.max_calls, oracle_rows=args.oracle_rows)
        atomic_tensor(args.output_dir/"witness.pt", payload)
        record.update(state=result["state"], result=result, witness_sha256=file_sha256(args.output_dir/"witness.pt"))
    except Exception as error:
        record.update(state="execution_failed", error=repr(error)); atomic(args.output_dir/"result.json", record)
        raise
    atomic(args.output_dir/"result.json", record)
    print(json.dumps({key: record[key] for key in ("state", "source", "anchor_sha256")}))
    return 0 if record["state"] in {"first_attention_difference", "no_attention_difference"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
