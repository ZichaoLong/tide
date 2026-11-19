"""Bounded CPU inspection of trusted saved training observations, never a model run.

The candidate is actual and baseline is reference. Original thresholds are fixed.
Private mmap mappings and bounded logical-order chunks avoid whole-tensor copies.
Scalar Decimal/FP64 AdamW replay uses observed adjacent states only; anchor/VJP
predictions are explicitly counterfactual and cannot replace a missing update.
"""
from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
from decimal import Decimal, localcontext
import gc
import hashlib
import heapq
import json
import math
import mmap
import os
from pathlib import Path
import re
import resource
import stat
import sys
import time

import torch

from scripts.job import atomic


TOLERANCE = {"atol": 1e-6, "rtol": 2e-3}
MAX_OUTPUT_BYTES = 64 << 20


def encoded(value):
    """JSON has no NaN/Infinity numbers; retain such observations as strings."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def coordinates(shape, index):
    result = []
    for size in reversed(shape):
        result.append(index % size)
        index //= size
    return list(reversed(result))


def flat_chunk(value, start, length):
    """A logical flat slice without flattening a whole noncontiguous tensor."""
    if value.layout != torch.strided or value.device.type != "cpu" or value.is_complex():
        raise ValueError("only real strided CPU tensors are supported")
    if value.is_contiguous():
        return value.detach().view(-1)[start:start + length]
    indices = torch.arange(start, start + length, dtype=torch.int64)
    parts = []
    for size in reversed(value.shape):
        parts.append(indices.remainder(size))
        indices = torch.div(indices, size, rounding_mode="floor")
    return value.detach()[tuple(reversed(parts))]


def scalar_at(value, index):
    if not isinstance(value, torch.Tensor) or not 0 <= index < value.numel():
        raise ValueError("coordinate is outside its tensor")
    return value[tuple(coordinates(value.shape, index))].item()


def support_field(name):
    last = name.rsplit(".", 1)[-1]
    return last == "strengths" or (
        (".state." in name or ".final_state." in name)
        and last in {"weight", "short_weight", "long_weight"})


def _top_indices(scores, limit):
    """Top finite scores, with deterministic lower-index tie breaking."""
    count = min(limit, int(torch.isfinite(scores).sum()))
    if not count:
        return []
    cutoff = torch.topk(scores, count).values[-1]
    greater = (scores > cutoff).nonzero().flatten().tolist()
    equal = (scores == cutoff).nonzero().flatten()[:count - len(greater)].tolist()
    return greater + equal


def compare_tensor(actual, reference, name, *, chunk_elements=262144, top_k=4,
                   check_budget=lambda: None):
    row = dict(field=name, shape=list(actual.shape), dtype=str(actual.dtype),
               reference_shape=list(reference.shape), reference_dtype=str(reference.dtype))
    if (actual.shape != reference.shape or
            actual.is_floating_point() != reference.is_floating_point()):
        return dict(row, metadata_failure=True, exact=False, top_coordinates=[])
    if actual.layout != torch.strided or reference.layout != torch.strided:
        raise ValueError("sparse/quantized observations require a separate diagnostic")
    changed = failures = nonfinite = support = 0
    max_abs = max_scaled = sum_delta_sq = sum_reference_sq = 0.0
    heap = []
    discrete_coordinates, support_coordinates, nonfinite_coordinates = [], [], []
    def point(index, a, b, score=None):
        result = dict(flat_index=index, index=coordinates(actual.shape, index),
                      actual=encoded(a), reference=encoded(b))
        if score is not None:
            result["scaled_error"] = encoded(score)
        return result
    for start in range(0, actual.numel(), chunk_elements):
        check_budget()
        length = min(chunk_elements, actual.numel() - start)
        a, b = flat_chunk(actual, start, length), flat_chunk(reference, start, length)
        different = a != b
        changed += int(different.sum())
        if not actual.is_floating_point():
            for index in different.nonzero().flatten()[:max(0, top_k - len(discrete_coordinates))].tolist():
                discrete_coordinates.append(point(start + index, a[index].item(), b[index].item()))
            continue  # int64 times/indices never pass through float64.
        a, b = a.double(), b.double()
        finite = torch.isfinite(a) & torch.isfinite(b)
        nonfinite += int((~finite).sum())
        for index in (~finite).nonzero().flatten()[:max(0, top_k - len(nonfinite_coordinates))].tolist():
            nonfinite_coordinates.append(point(start + index, a[index].item(), b[index].item()))
        if support_field(name):
            crossed = (a > 0) != (b > 0)
            support += int(crossed.sum())
            for index in crossed.nonzero().flatten()[:max(0, top_k - len(support_coordinates))].tolist():
                support_coordinates.append(point(start + index, a[index].item(), b[index].item()))
        # Nonfinite observations are a separate failure. Finite entries remain
        # useful witnesses, but do not certify a partially finite tensor.
        delta = torch.where(finite, a - b, 0.0)
        scaled = delta.abs() / (TOLERANCE["atol"] + TOLERANCE["rtol"] * torch.where(finite, b.abs(), 0.0))
        failures += int((scaled > 1).sum())
        if length:
            max_abs = max(max_abs, float(delta.abs().max()))
            max_scaled = max(max_scaled, float(scaled.max()))
        sum_delta_sq += float(delta.square().sum())
        sum_reference_sq += float(torch.where(finite, b, 0.0).square().sum())
        ranked = torch.where(finite & different, scaled, -math.inf)
        for index in _top_indices(ranked, top_k):
            score = float(scaled[index]); flat = start + index
            item = (score, -flat, a[index].item(), b[index].item())
            if len(heap) < top_k:
                heapq.heappush(heap, item)
            elif item[:2] > heap[0][:2]:
                heapq.heapreplace(heap, item)
    top = [point(-neg_index, a, b, score) for score, neg_index, a, b in sorted(heap, reverse=True)]
    return dict(row, elements=actual.numel(), different_elements=changed,
                exact=not (changed or nonfinite or actual.dtype != reference.dtype),
                dtype_difference=actual.dtype != reference.dtype,
                metadata_failure=bool(changed and not actual.is_floating_point()),
                threshold_failure=bool(failures and not nonfinite),
                finite_threshold_failure_elements=failures, nonfinite_elements=nonfinite,
                support_difference_elements=support, max_abs=encoded(max_abs),
                max_scaled_error=encoded(max_scaled),
                relative_l2=encoded(math.sqrt(sum_delta_sq) / max(math.sqrt(sum_reference_sq), 1e-30)),
                top_coordinates=top if actual.is_floating_point() else discrete_coordinates,
                support_coordinates=support_coordinates, nonfinite_coordinates=nonfinite_coordinates)


def _pairs(actual, reference, path="root"):
    if isinstance(actual, torch.Tensor) and isinstance(reference, torch.Tensor):
        yield path, actual, reference
    elif type(actual) is not type(reference):
        yield path + ".type", type(actual).__name__, type(reference).__name__
    elif is_dataclass(actual):
        for field in fields(actual):
            yield from _pairs(getattr(actual, field.name), getattr(reference, field.name), path + "." + field.name)
    elif isinstance(actual, dict):
        if actual.keys() != reference.keys():
            yield path + ".keys", sorted(map(str, actual)), sorted(map(str, reference))
        for key in sorted(actual.keys() & reference.keys(), key=str):
            yield from _pairs(actual[key], reference[key], path + "." + str(key))
    elif isinstance(actual, (tuple, list)):
        if len(actual) != len(reference):
            yield path + ".length", len(actual), len(reference)
        for index, (a, b) in enumerate(zip(actual, reference)):
            yield from _pairs(a, b, path + "." + str(index))
    elif actual != reference:
        yield path, actual, reference


def compare_saved(actual, reference, *, chunk_elements=262144, top_k=4,
                  max_fields=8192, check_budget=lambda: None):
    if not 1 <= chunk_elements <= 1048576 or not 1 <= top_k <= 32 or not 1 <= max_fields <= 16384:
        raise ValueError("invalid finite comparison bounds")
    result = dict(tolerance=dict(TOLERANCE), reference_direction="actual candidate minus reference baseline",
                  tensor_fields=0, exact_tensor_fields=0, changed_fields=[], metadata_failures=[],
                  threshold_failures=[], dtype_differences=[], support_differences=[], nonfinite=[],
                  omitted_changed_fields=0)
    for name, a, b in _pairs(actual, reference):
        check_budget()
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            row = compare_tensor(a, b, name, chunk_elements=chunk_elements, top_k=top_k, check_budget=check_budget)
            result["tensor_fields"] += 1
            result["exact_tensor_fields"] += int(row["exact"])
            for flag, key in [("metadata_failure", "metadata_failures"), ("threshold_failure", "threshold_failures"),
                              ("dtype_difference", "dtype_differences"), ("support_difference_elements", "support_differences"),
                              ("nonfinite_elements", "nonfinite")]:
                if row.get(flag):
                    result[key].append(name)
            if not row["exact"]:
                if len(result["changed_fields"]) < max_fields:
                    result["changed_fields"].append(row)
                else:
                    result["omitted_changed_fields"] += 1
        else:
            result["metadata_failures"].append(name)
            if len(result["changed_fields"]) < max_fields:
                result["changed_fields"].append(dict(field=name, metadata_failure=True, actual=repr(a), reference=repr(b)))
            else:
                result["omitted_changed_fields"] += 1
    result["state"] = ("nonfinite" if result["nonfinite"] else "metadata_failed" if result["metadata_failures"]
                       else "dtype_failed" if result["dtype_differences"] else "support_failed" if result["support_differences"]
                       else "finite_deviation" if result["threshold_failures"] else "within_threshold")
    result["exact"] = result["tensor_fields"] == result["exact_tensor_fields"] and not result["metadata_failures"]
    return result


def _finite(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("nonfinite scalar input")
    return value


def optimizer_options(group):
    if group.get("amsgrad") or group.get("maximize") or group.get("differentiable"):
        raise ValueError("this replay supports ordinary non-AMSGrad, non-maximizing AdamW")
    result = {name: _finite(group[name]) for name in ["lr", "eps", "weight_decay"]}
    result["betas"] = tuple(_finite(x) for x in group["betas"])
    if (len(result["betas"]) != 2 or any(not 0 <= x < 1 for x in result["betas"])
            or result["lr"] < 0 or result["eps"] <= 0 or result["weight_decay"] < 0):
        raise ValueError("unsupported AdamW scalar options")
    return result


def adamw_fp64(parameter, gradient, first, second, step, group):
    """Independent scalar binary64 expression; does not invoke Torch AdamW."""
    options = optimizer_options(group)
    p, g, m, v = map(_finite, [parameter, gradient, first, second])
    if type(step) is not int or step < 0 or v < 0:
        raise ValueError("invalid AdamW prior state")
    b1, b2 = options["betas"]
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    denominator = math.sqrt(v) / math.sqrt(1 - b2 ** (step + 1)) + options["eps"]
    p = p * (1 - options["lr"] * options["weight_decay"]) - options["lr"] * m / (1 - b1 ** (step + 1)) / denominator
    return {key: _finite(value) for key, value in dict(parameter=p, first_moment=m, second_moment=v, denominator=denominator).items()}


def adamw_decimal(parameter, gradient, first, second, step, group):
    """80-digit scalar equation from exact binary input values, without rounded intermediates."""
    options = optimizer_options(group)
    p, g, m, v = map(_finite, [parameter, gradient, first, second])
    if type(step) is not int or step < 0 or v < 0:
        raise ValueError("invalid AdamW prior state")
    with localcontext() as context:
        context.prec = 80
        p, g, m, v = map(Decimal.from_float, [p, g, m, v])
        b1, b2 = map(Decimal.from_float, options["betas"])
        lr, eps, decay = map(Decimal.from_float, [options["lr"], options["eps"], options["weight_decay"]])
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        denominator = (v / (1 - b2 ** (step + 1))).sqrt() + eps
        p = p * (1 - lr * decay) - lr * (m / (1 - b1 ** (step + 1))) / denominator
        values = dict(parameter=p, first_moment=m, second_moment=v, denominator=denominator)
        if not all(x.is_finite() for x in values.values()):
            raise ValueError("nonfinite Decimal result")
        return {name: str(value) for name, value in values.items()}


def _point(snapshot, parameter, index):
    slot = snapshot["optimizer"]["state"][parameter]
    step = slot["step"]
    if isinstance(step, torch.Tensor):
        if step.ndim:
            raise ValueError("AdamW step must be scalar")
        step = step.item()
    if not isinstance(step, (float, int)) or not math.isfinite(step) or step < 0 or step != int(step):
        raise ValueError("invalid observed AdamW step")
    p = snapshot["parameters"][parameter]
    if any(slot[key].shape != p.shape for key in ["exp_avg", "exp_avg_sq"]):
        raise ValueError("moment/parameter shape mismatch")
    return dict(parameter=_finite(scalar_at(p, index)), first=_finite(scalar_at(slot["exp_avg"], index)),
                second=_finite(scalar_at(slot["exp_avg_sq"], index)), step=int(step))


def replay_coordinate(before_reference, before_actual, current_reference, current_actual,
                      parameter, index, *, counterfactual=False):
    """Replay an observed adjacent step, or a clearly labelled anchor/VJP prediction."""
    b, a = _point(before_reference, parameter, index), _point(before_actual, parameter, index)
    gb = _finite(scalar_at(current_reference["gradients"][parameter], index))
    ga = _finite(scalar_at(current_actual["gradients"][parameter], index))
    group = before_reference["optimizer"]["group"]
    options = optimizer_options(group)
    if options != optimizer_options(before_actual["optimizer"]["group"]) or a["step"] != b["step"]:
        raise ValueError("owners have different optimizer scalar options or steps")
    observed = None
    if not counterfactual:
        observed = dict(reference=_point(current_reference, parameter, index), actual=_point(current_actual, parameter, index))
        for label, current, previous in [("reference", current_reference, b), ("actual", current_actual, a)]:
            if observed[label]["step"] != previous["step"] + 1:
                raise ValueError("saved observations are not adjacent AdamW updates")
            if optimizer_options(current["optimizer"]["group"]) != options:
                raise ValueError("optimizer options changed between saved updates")
    def evaluate(old, gradient):
        values = [old["parameter"], gradient, old["first"], old["second"], old["step"], group]
        return dict(decimal=adamw_decimal(*values), fp64=adamw_fp64(*values))
    predictions = dict(reference=evaluate(b, gb), actual=evaluate(a, ga))
    parameter_swapped = dict(b, parameter=a["parameter"])
    predictions["parameter_history_only"] = evaluate(parameter_swapped, gb)
    predictions["parameter_and_moment_history"] = evaluate(a, gb)
    predictions["gradient_only_from_reference_state"] = evaluate(b, ga)
    with localcontext() as context:
        context.prec = 80
        d = {key: Decimal(value["decimal"]["parameter"]) for key, value in predictions.items()}
        predicted_delta = d["actual"] - d["reference"]
        contributions = dict(previous_parameter=d["parameter_history_only"] - d["reference"],
                             previous_moments=d["parameter_and_moment_history"] - d["parameter_history_only"],
                             current_gradient=d["actual"] - d["parameter_and_moment_history"])
        observed_delta = None if observed is None else Decimal.from_float(observed["actual"]["parameter"]) - Decimal.from_float(observed["reference"]["parameter"])
        residual = None if observed_delta is None else observed_delta - predicted_delta
        gradient_delta = Decimal.from_float(ga) - Decimal.from_float(gb)
        gain = None if not gradient_delta else (d["actual"] - d["parameter_and_moment_history"]) / gradient_delta
        result = dict(parameter=parameter, flat_index=index,
            state="anchor_vjp_counterfactual" if counterfactual else "observed_adjacent_update_replay",
            before=dict(reference=b, actual=a), gradients=dict(reference=gb, actual=ga), options=options,
            observed_after=observed, predictions=predictions,
            predicted_parameter_difference=str(predicted_delta),
            actual_parameter_difference=None if observed_delta is None else str(observed_delta),
            observed_minus_predicted_difference=None if residual is None else str(residual),
            ordered_contributions={key: str(value) for key, value in contributions.items()},
            contribution_order="previous parameter, previous moments, current gradient; order-dependent algebra, not unique causal shares",
            current_gradient_secant_gain=None if gain is None else str(gain),
            gradient_magnitudes_at_most_eps=max(abs(ga), abs(gb)) <= options["eps"],
            epsilon_ratios={owner: dict(abs_gradient_over_eps=_finite(abs(gradient) / options["eps"]),
                                       abs_prior_first_moment_over_eps=_finite(abs(old["first"]) / options["eps"]),
                                       sqrt_prior_second_moment_over_eps=_finite(math.sqrt(old["second"]) / options["eps"]),
                                       denominator_over_eps=_finite(predictions[owner]["fp64"]["denominator"] / options["eps"]))
                            for owner, gradient, old in [("reference", gb, b), ("actual", ga, a)]},
            scope="Scalar equations on retained inputs; no backend intermediate rounding is simulated, no model or optimizer is executed")
        if observed is not None:
            result["observed_minus_decimal"] = {
                owner: {target: str(Decimal.from_float(observed[owner][source]) - Decimal(predictions[owner]["decimal"][target]))
                        for source, target in [("parameter", "parameter"), ("first", "first_moment"), ("second", "second_moment")]}
                for owner in ["reference", "actual"]}
        return result


def selected_coordinates(comparison, limit=64):
    choices = {}
    for row in comparison["changed_fields"]:
        prefix = next((p for p in ["root.parameters.", "root.gradients."] if row["field"].startswith(p)), None)
        if prefix is None:
            continue
        for point in row.get("top_coordinates", []):
            score = point.get("scaled_error")
            if not isinstance(score, (int, float)) or not math.isfinite(score):
                continue
            key = row["field"][len(prefix):], point["flat_index"]
            priority = (bool(row.get("threshold_failure")), score)
            if key not in choices or priority > choices[key]:
                choices[key] = priority
    ordered = sorted(choices, key=lambda key: (-choices[key][0], -choices[key][1], key))
    return ordered[:limit], len(ordered)


def anchor_bias_coordinates(anchor, reference, actual, *, top_k=4):
    """Selected old update-0 fields, ranked by counterfactual first-step change.

    The observed update-0 coordinate is unavailable. This does not pretend that
    VJP1 is the actual update-0 gradient, even when recorded repeated VJPs agree.
    """
    selected, rows = [], []
    for parameter in ["regional.26.second.bias", "regional.28.second.bias"]:
        if parameter not in anchor["parameters"]:
            rows.append(dict(parameter=parameter, state="field_absent"))
            continue
        size = anchor["parameters"][parameter].numel()
        if size > 4096:
            rows.append(dict(parameter=parameter, state="coordinate_budget_exceeded", elements=size))
            continue
        ranked = []
        for index in range(size):
            b = _point(anchor, parameter, index)
            gb = _finite(scalar_at(reference["gradients"][parameter], index))
            ga = _finite(scalar_at(actual["gradients"][parameter], index))
            predictions = [adamw_decimal(b["parameter"], g, b["first"], b["second"], b["step"], anchor["optimizer"]["group"])
                           for g in [gb, ga]]
            with localcontext() as context:
                context.prec = 80
                pb, pa = [Decimal(p["parameter"]) for p in predictions]
                scale = abs(pa - pb) / (Decimal.from_float(TOLERANCE["atol"]) + Decimal.from_float(TOLERANCE["rtol"]) * abs(pb))
                ranked.append((scale, index))
        chosen = sorted(ranked, key=lambda row: (-row[0], row[1]))[:top_k]
        selected.extend((parameter, index) for _, index in chosen)
        rows.append(dict(parameter=parameter, state="counterfactual_coordinates_ranked", elements=size,
                         selected=[dict(flat_index=index, predicted_scaled_parameter_difference=str(scale)) for scale, index in chosen],
                         scope="Ranked by anchor+VJP1 Decimal prediction; observed update-0 worst coordinates were not retained"))
    return selected, rows


class Inputs:
    def __init__(self, directory, record, maximum_bytes, verify_sha, check_budget):
        self.directory = Path(directory).resolve()
        self.manifest = {row["path"]: row for row in record["artifacts"]}
        self.maximum_bytes, self.verify_sha = maximum_bytes, verify_sha
        self.check_budget, self.identities = check_budget, {}

    def load(self, relative):
        self.check_budget()
        if relative not in self.manifest:
            raise ValueError("payload is absent from source manifest: " + relative)
        path = self.directory / relative
        if not path.resolve().is_relative_to(self.directory):
            raise ValueError("payload escaped its declared qualification directory")
        info = path.stat(); entry = self.manifest[relative]
        if not stat.S_ISREG(info.st_mode) or info.st_size != entry["bytes"]:
            raise ValueError("payload size/type differs from source manifest: " + relative)
        identity = dict(path=str(path), bytes=info.st_size, manifest_sha256=entry["sha256"],
                        device=info.st_dev, inode=info.st_ino, mtime_ns=info.st_mtime_ns,
                        content_sha256_verified=False)
        if relative in self.identities:
            previous = self.identities[relative]
            if any(identity[key] != previous[key] for key in ["bytes", "device", "inode", "mtime_ns"]):
                raise ValueError("input changed during diagnosis: " + relative)
        else:
            if sum(row["bytes"] for row in self.identities.values()) + info.st_size > self.maximum_bytes:
                raise MemoryError("declared unique input byte budget exceeded before mmap")
            if self.verify_sha:
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    while block := stream.read(8 << 20):
                        self.check_budget(); digest.update(block)
                if digest.hexdigest() != entry["sha256"]:
                    raise ValueError("input SHA256 differs from manifest: " + relative)
                identity["content_sha256_verified"] = True
            self.identities[relative] = identity
        # Explicit MAP_PRIVATE prevents accidental writes from reaching files.
        with torch.serialization.set_default_mmap_options(mmap.MAP_PRIVATE):
            return torch.load(path, map_location="cpu", mmap=True, weights_only=False)

    def validate_unchanged(self):
        for item in self.identities.values():
            info = Path(item["path"]).stat()
            if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != (
                    item["device"], item["inode"], item["bytes"], item["mtime_ns"]):
                raise ValueError("input changed during diagnosis")


def write_outputs(directory, report, witness, maximum=MAX_OUTPUT_BYTES):
    if not 0 < maximum <= MAX_OUTPUT_BYTES:
        raise ValueError("output budget must be positive and <=64 MiB")
    # Account the actual JSON encoding before either output is written.
    sizes = {name: len((json.dumps(data, indent=2, allow_nan=False) + "\n").encode())
             for name, data in [("result.json", report), ("witness.json", witness)]}
    if sum(sizes.values()) > maximum:
        raise MemoryError("bounded JSON/witness output budget exceeded; no output written")
    for name, data in [("witness.json", witness), ("result.json", report)]:
        atomic(Path(directory) / name, data)
    return sizes


def run(directory, output, *, reference="baseline", candidate="weight_access", chunk_elements=262144,
        top_k=4, coordinate_limit=64, maximum_input_gib=24, seconds=3600, verify_sha=False,
        output_bytes=MAX_OUTPUT_BYTES, maximum_rss_gib=16):
    if (not all(re.fullmatch(r"[a-z][a-z0-9_]*", name) for name in [reference, candidate])
            or reference == candidate or not 1 <= coordinate_limit <= 256
            or not 1 <= chunk_elements <= 1048576 or not 1 <= top_k <= 32
            or not math.isfinite(maximum_input_gib) or not 0 < maximum_input_gib <= 64
            or not math.isfinite(maximum_rss_gib) or not 0 < maximum_rss_gib <= 64
            or not math.isfinite(seconds) or not 0 < seconds <= 7200):
        raise ValueError("invalid finite diagnostic limits")
    output = Path(output)
    if output.exists():
        raise ValueError("choose a new output directory")
    if output.resolve().is_relative_to(Path(directory).resolve()):
        raise ValueError("outputs must be outside the frozen qualification inputs")
    if not 0 < output_bytes <= MAX_OUTPUT_BYTES:
        raise ValueError("output budget must be positive and <=64 MiB")
    source_path = Path(directory) / "result.json"
    source_bytes = source_path.read_bytes(); original = json.loads(source_bytes)
    if original.get("schema") != "tide-execution-qualification-v1" or original.get("precision") != "fp32" or original.get("tolerance") != TOLERANCE:
        raise ValueError("requires the original FP32 execution qualification and unchanged thresholds")
    cases = {case["name"]: case for case in original["candidates"]}
    if reference != "baseline" or candidate not in cases:
        raise ValueError("source record must contain this candidate against baseline")
    started = time.monotonic()
    def peak_rss():
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == "darwin" else value * 1024)
    def check_budget():
        if time.monotonic() - started > seconds:
            raise TimeoutError("finite saved-observation diagnostic time budget exhausted")
        if peak_rss() > maximum_rss_gib * 2**30:
            raise MemoryError("process RSS high-water budget exceeded")
    inputs = Inputs(directory, original, int(maximum_input_gib * 2**30), verify_sha, check_budget)
    report = dict(schema="tide-saved-training-diagnosis-v1", state="running", original_state=original["state"],
        original_qualified=original.get("qualified"), original_source=original["source"],
        source_record=dict(path=str(source_path.resolve()), sha256=hashlib.sha256(source_bytes).hexdigest(), bytes=len(source_bytes)),
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), torch=str(torch.__version__),
        tolerance=dict(TOLERANCE), reference=reference, candidate=candidate, comparisons=[],
        bounds=dict(chunk_elements=chunk_elements, top_coordinates_per_field=top_k, adamw_coordinates=coordinate_limit,
                    unique_input_bytes=int(maximum_input_gib * 2**30), output_bytes=output_bytes,
                    process_peak_rss_bytes=int(maximum_rss_gib * 2**30),
                    wall_seconds=seconds, maximum_simultaneous_mmaps=4),
        interpretation="Completed read-only diagnosis is not restored-update, formula or execution qualification; original failures remain",
        update0_presence={owner: (Path(directory) / owner / "update-0.pt").is_file() for owner in [reference, candidate]},
        missing_update0="This finite plan consumes update-1/2 only. Update-1 historical replay needs both update-0 payloads; presence is recorded separately. Anchor/VJP predictions are not observed update-0.")
    witness = dict(schema="tide-saved-training-coordinate-witness-v1", phases=[],
                   scope="Only selected scalar coordinates and optimizer context; no complete tensors, models or historical trajectory")
    for phase in ["vjp1", "update-1", "update-2"]:
        reference_data = inputs.load(reference + "/" + phase + ".pt")
        actual_data = inputs.load(candidate + "/" + phase + ".pt")
        comparison = compare_saved(actual_data, reference_data, chunk_elements=chunk_elements, top_k=top_k,
                                   check_budget=check_budget)
        old_check = (cases[candidate]["checks"]["vjp1_vs_baseline"] if phase == "vjp1" else
                     next(row for row in cases[candidate]["updates"] if row["index"] == int(phase[-1]))["vs_baseline"])
        old_failures = old_check["comparison"]["threshold_failures"]
        comparison["original_threshold_failure_fields_match"] = set(old_failures) == set(comparison["threshold_failures"])
        comparison["original_threshold_failure_count"] = len(old_failures)
        comparison["original_failure_field_sets_match"] = {
            key: {value["tensor"] if isinstance(value, dict) else value for value in old_check["comparison"][key]} == set(comparison[key])
            for key in ["threshold_failures", "metadata_failures", "dtype_differences", "support_differences", "nonfinite"]}
        selected, available = selected_coordinates(comparison, coordinate_limit)
        phase_witness = dict(phase=phase, comparison_coordinates=[], adamw=[], eligible_coordinates=available,
                             selected_coordinates=len(selected), selection="failing parameter/gradient fields first, then descending scaled error")
        for row in comparison["changed_fields"]:
            chosen = {key: row.pop(key) for key in ["top_coordinates", "support_coordinates", "nonfinite_coordinates"] if key in row}
            phase_witness["comparison_coordinates"].append(dict(field=row["field"], **chosen))
        if phase == "vjp1":
            anchor = inputs.load("anchor.pt")
            before_reference = before_actual = anchor["training"]
            bias_selected, bias_selection = anchor_bias_coordinates(before_reference, reference_data, actual_data, top_k=top_k)
            selected = list(dict.fromkeys(bias_selected + selected))[:coordinate_limit]
            phase_witness["anchor_bias_selection"] = bias_selection
            phase_witness["selected_coordinates"] = len(selected)
        elif phase == "update-2":
            before_reference = inputs.load(reference + "/update-1.pt")
            before_actual = inputs.load(candidate + "/update-1.pt")
        else:
            before_reference = before_actual = None
        for parameter, index in selected:
            check_budget()
            if phase == "update-1":
                row = dict(parameter=parameter, flat_index=index, state="missing_previous_update",
                           absent_update0_owners=[owner for owner, exists in report["update0_presence"].items() if not exists],
                           reason="No complete update-0 pair is consumed by this plan; anchor is two steps earlier, not an adjacent state")
            else:
                try:
                    row = replay_coordinate(before_reference, before_actual, reference_data, actual_data,
                                            parameter, index, counterfactual=phase == "vjp1")
                except (KeyError, ValueError, TypeError, OverflowError) as error:
                    row = dict(parameter=parameter, flat_index=index, state="replay_unavailable", reason=str(error))
            phase_witness["adamw"].append(row)
        report["comparisons"].append(dict(phase=phase, **comparison))
        witness["phases"].append(phase_witness)
        del reference_data, actual_data, before_reference, before_actual
        if phase == "vjp1":
            del anchor
        gc.collect()
    inputs.validate_unchanged()
    report.update(state="diagnosis_complete", elapsed_seconds=time.monotonic() - started,
                  process_peak_rss_bytes=peak_rss(),
                  input_files=list(inputs.identities.values()),
                  all_original_threshold_field_sets_match=all(row["original_threshold_failure_fields_match"] for row in report["comparisons"]),
                  all_original_failure_field_sets_match=all(all(row["original_failure_field_sets_match"].values()) for row in report["comparisons"]))
    witness["source_record"] = report["source_record"]
    write_outputs(output, report, witness, output_bytes)
    return report, witness


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate", default="weight_access")
    parser.add_argument("--chunk-elements", type=int, default=262144)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--coordinate-limit", type=int, default=64)
    parser.add_argument("--max-input-gib", type=float, default=24)
    parser.add_argument("--max-rss-gib", type=float, default=16)
    parser.add_argument("--seconds", type=float, default=3600)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--verify-input-sha256", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.threads <= 8:
        parser.error("threads must be 1..8")
    torch.set_num_threads(args.threads)
    report, _ = run(args.qualification_dir, args.output_dir, candidate=args.candidate,
                    chunk_elements=args.chunk_elements, top_k=args.top_k, coordinate_limit=args.coordinate_limit,
                    maximum_input_gib=args.max_input_gib, maximum_rss_gib=args.max_rss_gib,
                    seconds=args.seconds, verify_sha=args.verify_input_sha256)
    print(json.dumps({key: report[key] for key in ["state", "elapsed_seconds", "all_original_failure_field_sets_match"]}))
    return 0 if report["all_original_failure_field_sets_match"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
