"""Bounded scalar routing witnesses for two regional execution paths.

Two untimed, zero-state requests share one immutable parameter owner. Only
received mass, descriptor score, regional score, control and original int64
time are captured. Local allocation replays use these recorded inputs; they
do not rerun the model or change its trajectory. This is a forward diagnostic,
not a training qualification or a performance measurement.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
import hashlib
import importlib
import io
import json
import os
from pathlib import Path

import torch

from tide import ModelConfig, TideModel
from tide.precision import precision_context, precision_policy
from benchmarks.inference import (InferenceSchedule, OutputDiagnostic,
                                  bounded_parameter_identity, run_request,
                                  source_identity)
from benchmarks.workloads import paired_input, tensor_hash
from scripts.diagnose_bf16 import decimal_quad
from scripts.job import atomic
from scripts.qualify_precision import THRESHOLDS, atomic_tensor


STAGES = ("received", "raw_score", "score", "control")
MIB = 1 << 20


class CaptureLimit(RuntimeError):
    pass


class CaptureBudget:
    """Shared byte bound, checked before each detached host allocation."""
    def __init__(self, maximum):
        if type(maximum) is not int or maximum < 1:
            raise ValueError("capture bytes must be a positive integer")
        self.maximum, self.used = maximum, 0

    def copy(self, tensor):
        size = tensor.numel() * tensor.element_size()
        if self.used + size > self.maximum:
            raise CaptureLimit(f"scalar capture exceeds {self.maximum} bytes")
        self.used += size
        return tensor.detach().to("cpu", copy=True).contiguous()


class RoutingTrace:
    """Normalize serial/batched regions without retaining content or history.

    Install only around forward calls. In training, remove before backward so
    checkpoint recomputation does not masquerade as an additional logical call.
    No placed/concurrent model instrumentation is supported here.
    """
    def __init__(self, model, budget=None):
        if hasattr(model, "_placement_spec"):
            raise ValueError("routing capture currently requires one unplaced owner")
        self.model = model
        self.budget = budget if budget is not None else CaptureBudget(64 * MIB)
        self.entries, self.calls = {}, []
        self.current, self.call, self.phase = (), -1, "forward"

    def put(self, region, stage, tensor):
        row = self.entries[self.call, region]
        if stage in row:
            raise RuntimeError(f"duplicate routing capture: {self.call}/{region}/{stage}")
        if tensor.ndim != 3 or tensor.shape[0] != len(row["nodes"]):
            raise ValueError(f"routing scalar must be [node,batch,time]: {stage}")
        if tensor.shape[-1] != row["time"].numel():
            raise ValueError("routing time axis does not match the original int64 clock")
        row[stage] = self.budget.copy(tensor)

    def group(self, stage, tensor):
        offset = 0
        for region in self.current:
            count = len(self.model.graph.regions[region])
            self.put(region, stage, tensor[offset:offset+count])
            offset += count
        if offset != tensor.shape[0]:
            raise ValueError("routing node split did not exhaust the submitted axis")

    @contextmanager
    def region(self, regions, time):
        if time.dtype != torch.int64 or time.ndim != 1:
            raise TypeError("routing capture requires the original int64 time vector")
        previous = self.current
        self.current = tuple(regions)
        try:
            for region in regions:
                key = self.call, region
                if key in self.entries:
                    raise RuntimeError("one logical call evaluated a region more than once")
                nodes = tuple(self.model.graph.regions[region])
                self.entries[key] = dict(nodes=nodes, group=tuple(regions),
                    layer=self.model.graph.nodes[nodes[0]].layer, time=self.budget.copy(time),
                    fixed=self.model.region_group[region] < 0)
            yield
        finally:
            self.current = previous

    @contextmanager
    def install(self):
        model = self.model
        batch_module = importlib.import_module("tide.region_batch")
        original_run, original_receive = model._run, model._receive
        original_single, original_score = model._evaluate_region, model._score_regions
        original_batch = batch_module.evaluate_regions
        handles = []
        active = True

        def run(value, state=None, **kwargs):
            self.call += 1
            self.calls.append(dict(index=self.call, phase=self.phase,
                state_position=0 if state is None else state.position,
                batch=value.shape[0], positions=value.shape[1]))
            return original_run(value, state, **kwargs)

        def receive(*args, **kwargs):
            result = original_receive(*args, **kwargs)
            if active and self.current:
                self.group("received", result[0])
            return result

        def single(region, contents, masses, classes, time, *args, **kwargs):
            if not active:
                return original_single(region, contents, masses, classes, time, *args, **kwargs)
            with self.region((region,), time):
                result = original_single(region, contents, masses, classes, time, *args, **kwargs)
                self.put(region, "control", result[1])
                return result

        def batch(owner, regions, contents, masses, classes, time, *args, **kwargs):
            if not active or owner is not model:
                return original_batch(owner, regions, contents, masses, classes, time, *args, **kwargs)
            with self.region(regions, time):
                result = original_batch(owner, regions, contents, masses, classes, time, *args, **kwargs)
                for region, control in zip(regions, result[1]):
                    self.put(region, "control", control)
                return result

        def score(regions, z, raw, mass):
            result = original_score(regions, z, raw, mass)
            if active:
                for index, region in enumerate(regions):
                    self.put(region, "score", result[index])
            return result

        def describe(module, args, result):
            if self.current:
                self.group("raw_score", result[1])

        def serial_score(module, args, result):
            if len(self.current) == 1:
                self.put(self.current[0], "score", result)

        model._run, model._receive = run, receive
        model._evaluate_region, model._score_regions = single, score
        batch_module.evaluate_regions = batch
        handles.append(model.describe.register_forward_hook(describe))
        handles.extend(network.register_forward_hook(serial_score) for network in model.regional)
        try:
            yield self
        finally:
            # Non-reentrant checkpoint retains wrapper callables for backward.
            # Those closures must become transparent after forward capture.
            active = False
            model._run, model._receive = original_run, original_receive
            model._evaluate_region, model._score_regions = original_single, original_score
            batch_module.evaluate_regions = original_batch
            for handle in handles:
                handle.remove()

    def validate(self):
        for key, row in self.entries.items():
            required = ("received", "control") if row["fixed"] else STAGES
            if any(name not in row for name in required):
                raise RuntimeError(f"incomplete routing capture at {key}")
        expected = len(self.calls) * len(self.model.graph.regions)
        if len(self.entries) != expected:
            raise RuntimeError(f"captured {len(self.entries)} region calls; expected {expected}")

    def summary(self):
        digest = hashlib.sha256()
        full = candidates = 0
        for key, row in sorted(self.entries.items()):
            digest.update(str(key).encode())
            digest.update(str(row["nodes"]).encode())
            for name in ("time", *STAGES):
                if name in row:
                    digest.update(name.encode())
                    digest.update(str(tuple(row[name].shape)).encode())
                    digest.update(str(row[name].dtype).encode())
                    digest.update(row[name].view(torch.uint8).numpy().tobytes())
            full += int((row["control"] > 0).sum())
            candidates += int((row["received"] > 0).sum())
        return dict(calls=self.calls, region_calls=len(self.entries), full=full,
                    candidates=candidates, scalar_sha256=digest.hexdigest())


def difference(reference, candidate, tolerance):
    """Keep integer metadata exact; never subtract int64 clocks as FP64."""
    same_shape, same_dtype = reference.shape == candidate.shape, reference.dtype == candidate.dtype
    result = dict(shape_matches=same_shape, dtype_matches=same_dtype,
                  reference_dtype=str(reference.dtype), candidate_dtype=str(candidate.dtype))
    if not same_shape:
        return dict(result, reference_shape=list(reference.shape), candidate_shape=list(candidate.shape))
    unequal = reference != candidate
    result.update(exact=same_dtype and not bool(unequal.any()), different=int(unequal.sum()),
                  elements=reference.numel())
    if not reference.is_floating_point() or not candidate.is_floating_point():
        return result
    a, b = reference.double(), candidate.double()
    delta = (a-b).abs()
    finite = torch.isfinite(a) & torch.isfinite(b)
    result.update(finite=bool(finite.all()), nonfinite=int((~finite).sum()),
        max_abs=float(delta.max()) if delta.numel() and bool(finite.all()) else None,
        threshold_failures=int(((delta > tolerance["atol"] + tolerance["rtol"]*a.abs()) | ~finite).sum()))
    return result


def coordinate(key, row, index):
    node, sample, position = (int(x) for x in index)
    return dict(call=key[0], region=key[1], layer=row["layer"], node=row["nodes"][node],
                local_node=node, sample=sample, time_index=position,
                original_time=int(row["time"][position]))


def compare_traces(reference, candidate, tolerance, max_witnesses=4):
    if type(max_witnesses) is not int or not 1 <= max_witnesses <= 16:
        raise ValueError("max witnesses must be 1..16")
    a, b = reference.entries, candidate.entries
    metadata = []
    if reference.calls != candidate.calls:
        metadata.append(dict(field="calls", reference=reference.calls, candidate=candidate.calls))
    if a.keys() != b.keys():
        metadata.append(dict(field="region_keys", reference_only=sorted(a.keys()-b.keys()),
                             candidate_only=sorted(b.keys()-a.keys())))
    stages, first, witnesses = [], {}, []
    support_total = 0
    witnessed_rows = set()
    for key in sorted(a.keys() & b.keys()):
        left, right = a[key], b[key]
        if left["nodes"] != right["nodes"] or left["fixed"] != right["fixed"]:
            metadata.append(dict(key=key, field="nodes_or_fixed"))
            continue
        clock = difference(left["time"], right["time"], tolerance)
        if not clock.get("exact", False):
            metadata.append(dict(key=key, field="original_int64_time", difference=clock))
        for stage in STAGES:
            if stage not in left and stage not in right:
                continue
            if stage not in left or stage not in right:
                metadata.append(dict(key=key, field=stage, missing=True)); continue
            values = difference(left[stage], right[stage], tolerance)
            row = dict(key=key, stage=stage, difference=values)
            if not values.get("exact", False) and stage not in first:
                first[stage] = row.copy()
                if values["shape_matches"]:
                    indices = (left[stage] != right[stage]).nonzero()
                    if indices.numel():
                        first[stage]["coordinate"] = coordinate(key, left, indices[0])
            if not values["shape_matches"] or not values["dtype_matches"]:
                metadata.append(dict(key=key, field=stage, difference=values))
            if stage == "control" and values["shape_matches"]:
                changed = (left[stage] > 0) != (right[stage] > 0)
                count = int(changed.sum())
                row["support_differences"] = count
                support_total += count
                if count and len(witnesses) < max_witnesses:
                    for index in changed.nonzero().tolist():
                        # One complete regional row explains every changed node
                        # at this sample/time; do not duplicate its payload.
                        row_key = (*key, index[1], index[2])
                        if row_key in witnessed_rows:
                            continue
                        witnesses.append(coordinate(key, left, index))
                        witnessed_rows.add(row_key)
                        if len(witnesses) == max_witnesses:
                            break
            stages.append(row)
    return dict(metadata_failures=metadata, support_differences=support_total,
        nonfinite_elements=sum(row["difference"].get("nonfinite", 0) for row in stages),
        threshold_failure_elements=sum(row["difference"].get("threshold_failures", 0) for row in stages),
        first_changed_stage=first, witnesses=witnesses, stages=stages,
        ordering="logical model call, then original region order; first recorded scalar per stage, not an upstream causal proof",
        tolerance=dict(tolerance), support_policy="exact positive support and original int64 time; never relaxed by value tolerance")


def allocation_payload(trace, key):
    selected = trace.entries[key]
    group = selected["group"]
    rows = [trace.entries[key[0], region] for region in group]
    payload = dict(group=group, selected_index=group.index(key[1]), fixed=selected["fixed"],
                   time=selected["time"], nodes=selected["nodes"])
    for name in ("received", "score", "control"):
        if name in selected:
            payload[name] = torch.stack([row[name] for row in rows])
    return payload


def replay_allocation(payload, roles, budget, profile, partition, *, sample, position,
                      device="cpu", precision="fp32", tolerance=None):
    """Same rounded inputs in original and single-region shapes, plus Decimal.

    Replaying changes only submission shape, never scores supplied to the model.
    The recorded shape replay must match capture before it can explain that
    capture. FP64 is used for one CPU row only.
    """
    from tide.region import region_control
    tolerance = THRESHOLDS[precision] if tolerance is None else tolerance
    index = payload["selected_index"]
    received = payload["received"]
    r, n, b, length = received.shape
    actual = payload["control"][index, :, sample, position]
    original_received = received[index, :, sample, position]
    if payload["fixed"]:
        exact = torch.tensor([min(1.0, 2*float(x)) for x in original_received], dtype=torch.float64)
        return dict(kind="fixed_control", received=original_received.tolist(), actual=actual.tolist(),
            independent=exact.tolist(), support_matches_independent=torch.equal(actual > 0, exact > 0))
    if profile != "quad":
        raise ValueError("independent routing witness currently requires quad allocation")
    score = payload["score"]
    scalar_score = score[index, :, sample, position]
    oracle = decimal_quad(scalar_score.tolist(), original_received.tolist(), roles, budget, partition)
    role_tensor = torch.tensor(roles, dtype=torch.bool, device=device)
    def fold(value):
        return value.permute(1, 0, 2, 3).reshape(n, r*b, length).to(device)
    with torch.no_grad(), precision_context(device, precision):
        grouped = region_control(fold(score), fold(received), role_tensor, budget, profile, partition)
        grouped = grouped.reshape(n, r, b, length).permute(1, 0, 2, 3).cpu()
        single = region_control(score[index].to(device), received[index].to(device), role_tensor,
                                budget, profile, partition).cpu()
    with torch.no_grad():
        double = region_control(scalar_score.double()[:, None, None],
            original_received.double()[:, None, None], torch.tensor(roles, dtype=torch.bool),
            budget, profile, partition)[:, 0, 0]
    grouped_row, single_row = grouped[index, :, sample, position], single[:, sample, position]
    decimal_values = torch.tensor(oracle["control"], dtype=torch.float64)
    return dict(kind="quad", original_submission_shape=[n, r*b, length],
        single_region_shape=[n, b, length], score=scalar_score.tolist(),
        received=original_received.tolist(), actual=actual.tolist(), independent_decimal=oracle,
        same_shape_replay=difference(payload["control"], grouped, tolerance),
        single_vs_grouped=difference(grouped[index], single, tolerance),
        row_grouped=grouped_row.tolist(), row_single=single_row.tolist(),
        cpu_fp64_row=double.tolist(),
        support_matches_decimal=torch.equal(actual > 0, decimal_values > 0),
        cpu_fp64_support_matches_decimal=torch.equal(double > 0, decimal_values > 0),
        grouped_vs_single_support_differences=int(((grouped[index] > 0) != (single > 0)).sum()))


def historical_record(path):
    record = json.loads(path.read_text())
    tide = record["subjects"]["tide"]
    return dict(path=str(path), file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source=record["source"], config=ModelConfig.from_dict(tide["spec"]).to_dict(),
        schedule=record["schedule"], seed=record["matching_reference"]["seed"],
        precision=record["precision"]["name"], input_shape=tide["input_shape"],
        parameters_sha256=tide["initial_parameters"]["sha256"], input_sha256=tide["input_sha256"],
        output_support_sha256=tide["diagnostics"]["output_support_sha256"],
        original_output_times_sha256=tide["diagnostics"]["original_output_times_sha256"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--region-batch-sizes", type=int, nargs=2, default=(1, 4))
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--prefill", type=int, default=128)
    parser.add_argument("--prefill-chunk", type=int, default=32)
    parser.add_argument("--decode", type=int, default=16)
    parser.add_argument("--seed", type=int, default=751)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--capture-mib", type=int, default=64)
    parser.add_argument("--output-budget-mib", type=int, default=256)
    parser.add_argument("--max-witnesses", type=int, default=4)
    parser.add_argument("--historical-results", type=Path, nargs=2,
                        help="old inference JSONs in the same path order; require matching config/parameter/input identities")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    directory = args.output.with_suffix(".data")
    if args.output.exists() or directory.exists():
        parser.error("choose a new output and sibling .data directory")
    if min(args.batch, args.threads, *args.region_batch_sizes) < 1:
        parser.error("batch, threads and region batch sizes must be positive")
    if not 1 <= args.capture_mib <= 192 or not 1 <= args.output_budget_mib <= 256:
        parser.error("capture bound must be 1..192 MiB; output bound 1..256 MiB")
    if not 1 <= args.max_witnesses <= 16:
        parser.error("max witnesses must be 1..16")
    config = ModelConfig.from_dict(json.loads(args.config.read_text()))
    if config.topology.timing != "settle" or config.allocation != "quad":
        parser.error("this CLI selects Settle/quad forward routing only")
    schedule = InferenceSchedule(args.prefill, args.prefill_chunk, args.decode)
    historical = [historical_record(path) for path in args.historical_results] if args.historical_results else []
    for size, old in zip(args.region_batch_sizes, historical):
        if (old["config"] != replace(config, region_batch_size=size).to_dict()
                or old["schedule"] != asdict(schedule) or old["seed"] != args.seed
                or old["precision"] != args.precision
                or old["input_shape"] != [args.batch, schedule.positions, config.input_dim or config.d]):
            parser.error("historical config/schedule/seed/precision/input shape differs from the requested witness")
    torch.set_num_threads(args.threads)
    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401 -- admission belongs to the caller
        torch.npu.set_device(args.device)
        torch.npu.matmul.allow_hf32 = False
        torch.npu.conv.allow_hf32 = False
    directory.mkdir(parents=True)
    record = dict(schema="tide-routing-divergence-v1", state="running", source=source_identity(),
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        arguments={key: str(value) if isinstance(value, Path) else
                   [str(x) for x in value] if key == "historical_results" and value else value
                   for key, value in vars(args).items()},
        torch=str(torch.__version__), cpu_affinity=sorted(os.sched_getaffinity(0)),
        precision=precision_policy(args.device, args.precision), thresholds=THRESHOLDS[args.precision],
        historical=historical, paths=[],
        scope="two intrusive forward requests, same read-only parameter owner, fresh private state per path; no optimizer, training qualification or timing",
        unproven=["scalar captures do not establish the earliest vector/operator cause of changed descriptor or regional scores",
                  "independent Decimal explains rounded allocation inputs only; it does not certify whole-model trajectories",
                  "instrumentation synchronizes host copies and can alter device scheduling; historical output hashes are checked separately"])
    atomic(args.output, record)
    try:
        torch.manual_seed(args.seed)
        model = TideModel(config).eval()
        identity = bounded_parameter_identity(model)
        if not identity["finite"]:
            raise ArithmeticError("nonfinite initial parameters")
        x_cpu = paired_input(args.batch, schedule.positions, config.input_dim or config.d, args.seed+1)
        input_identity = tensor_hash(x_cpu)
        record.update(parameters=identity, input_sha256=input_identity, schedule=asdict(schedule))
        for old in historical:
            if identity["sha256"] != old["parameters_sha256"] or input_identity != old["input_sha256"]:
                raise ValueError("historical parameter/input SHA mismatch; no model request executed")
        model = model.to(args.device)
        value = x_cpu.to(args.device); del x_cpu
        budget, traces = CaptureBudget(args.capture_mib*MIB), []
        for size in args.region_batch_sizes:
            model.config = replace(config, region_batch_size=size)
            trace, output = RoutingTrace(model, budget), OutputDiagnostic()
            current = dict(region_batch_size=size, config=model.config.to_dict(), state="running")
            record["paths"].append(current); atomic(args.output, record)
            def observe(readout, raw, phase, state):
                output(readout, raw, phase, state)
                current["completed_calls"] = len(trace.calls)
                current["captured_bytes_total"] = budget.used
                atomic(args.output, record)
            with trace.install():
                state, _ = run_request(model, value, schedule, precision=args.precision,
                                       observer=observe, timing=False)
            trace.validate()
            current.update(state="completed", routing=trace.summary(), output=output.report(state))
            del state
            traces.append(trace); atomic(args.output, record)
        compared = compare_traces(*traces, THRESHOLDS[args.precision], args.max_witnesses)
        record["comparison"] = compared
        payloads, explained, submission_cache = [], [], {}
        for selected in compared["witnesses"]:
            key = selected["call"], selected["region"]
            nodes = model.graph.regions[key[1]]
            roles = [model.graph.nodes[node].role == "H" for node in nodes]
            budget_value = config.budget or max(1, model.graph.config.n//32)
            payload = []
            for path, trace in enumerate(traces):
                group = trace.entries[key]["group"]
                cache_key = path, key[0], group
                if cache_key not in submission_cache:
                    submission_cache[cache_key] = allocation_payload(trace, key)
                # Reuse tensor storage when multiple witnesses select different
                # sample/time rows in the same original allocation submission.
                payload.append(dict(submission_cache[cache_key], selected_index=group.index(key[1]),
                                    nodes=trace.entries[key]["nodes"]))
            rows = [replay_allocation(item, roles, budget_value, config.allocation,
                     model.graph.config.partition, sample=selected["sample"], position=selected["time_index"],
                     device=args.device, precision=args.precision) for item in payload]
            oracle_matches = all(row.get("support_matches_decimal", row.get("support_matches_independent", False)) for row in rows)
            replay_matches = all(row.get("same_shape_replay", {}).get("exact", True) for row in rows)
            explained.append(dict(coordinate=selected, roles=roles, budget=budget_value, paths=rows,
                finding=("recorded allocation inputs cross the independent support boundary" if oracle_matches else
                         "recorded production support differs from the independent rounded-input solution"),
                original_shape_replay_exact=replay_matches))
            payloads.append(dict(coordinate=selected, roles=roles, budget=budget_value,
                profile=config.allocation, partition=model.graph.config.partition, paths=payload))
        record["allocation_witnesses"] = explained
        if payloads:
            # Only original allocation submission rows; no model/checkpoint/history dump.
            with io.BytesIO() as stream:
                torch.save(payloads, stream)
                payload_bytes = stream.tell()
            json_bytes = len(json.dumps(record, indent=2).encode()) + 4096
            if payload_bytes + json_bytes > args.output_budget_mib*MIB:
                raise CaptureLimit("minimal witness plus JSON exceeds the declared output budget")
            atomic_tensor(directory/"allocation-witnesses.pt", payloads)
        for current, old in zip(record["paths"], historical):
            current["historical_output_identity"] = {
                name: current["output"][name] == old[name]
                for name in ("output_support_sha256", "original_output_times_sha256")}
        finite_outputs = all(path["output"]["finite_outputs"] and path["output"]["finite_private_state"]
                             for path in record["paths"])
        record.update(state="completed", captured_bytes=budget.used,
            outcome="nonfinite" if compared["nonfinite_elements"] or not finite_outputs else
                    "metadata_difference" if compared["metadata_failures"] else
                    "support_difference_observed" if compared["support_differences"] else
                    "finite_value_deviation" if compared["threshold_failure_elements"] else "no_support_difference_observed",
            historical_witness_reproduced=(all(all(path["historical_output_identity"].values())
                                               for path in record["paths"]) if historical else None))
        projected = len(json.dumps(record, indent=2).encode()) + 1024 + sum(p.stat().st_size for p in directory.iterdir())
        if projected > args.output_budget_mib*MIB:
            raise CaptureLimit("minimal witness plus JSON exceeds the declared output budget")
        record["output_bytes_before_final_record"] = projected
        atomic(args.output, record)
        return record
    except Exception as error:
        if isinstance(error, CaptureLimit):
            # A budget failure retains a small durable status, not an oversized
            # incomplete report that would itself breach the declared bound.
            record.pop("comparison", None)
            record.pop("allocation_witnesses", None)
        record.update(state="execution_failed", error=repr(error))
        atomic(args.output, record)
        raise


if __name__ == "__main__":
    main()
