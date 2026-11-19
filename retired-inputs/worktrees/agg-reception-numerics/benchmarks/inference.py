"""Bounded backbone prefill, real incremental decode and chunked streaming.

Each request starts at zero private state, continues KV/SSD/pending without an
intermediate Timed flush, and drains once at the end. Outputs are materialized,
synchronized and released. The older continuous.run_stream contract is unchanged.
There is no task head, token sampling, service queue or language-model TTFT.
"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import statistics
import subprocess
import time

import torch

from benchmarks.baselines import TransformerBaseline, parameter_count
from benchmarks.continuous import _tensors, forward_state, state_memory, synchronize
from benchmarks.matching import match_transformers
from benchmarks.streaming_audit import FrozenParameterInventory, StreamingWorkAudit
from benchmarks.transformer_placement import LayerPlacedTransformer
from benchmarks.workloads import paired_input, tensor_hash
from tide import ModelConfig, TideModel
from tide.precision import precision_context, precision_policy


@dataclass(frozen=True)
class InferenceSchedule:
    prefill: int = 128
    prefill_chunk: int = 128
    decode_steps: int = 256
    stream_positions: int = 0
    stream_chunk: int = 128

    def __post_init__(self):
        for name in ("prefill", "prefill_chunk", "stream_chunk"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("decode_steps", "stream_positions"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")

    @property
    def positions(self):
        return self.prefill+self.decode_steps+self.stream_positions

    def calls(self):
        for start in range(0, self.prefill, self.prefill_chunk):
            yield "prefill", start, min(self.prefill, start+self.prefill_chunk)
        stop = self.prefill+self.decode_steps
        for start in range(self.prefill, stop):
            yield "decode", start, start+1
        for start in range(stop, self.positions, self.stream_chunk):
            yield "stream", start, min(self.positions, start+self.stream_chunk)


def execution_devices(model, x):
    devices = list(getattr(model, "_placement_devices", ()))
    devices += [x.device, *(p.device for p in model.parameters())]
    return tuple(dict.fromkeys(torch.device(d) for d in devices))


def synchronize_all(devices):
    for device in devices:
        synchronize(device)


def _prompt_ready(model, state_position, prefill, initial_position=0):
    if isinstance(model, TideModel) and not model.settle:
        # A Timed edge consumes one tick, including long-distance spatial edges.
        # The depth bound also covers the longest possible terminal path. This
        # is structural output readiness, never attribution of later memories.
        last_tick = (initial_position+prefill-1)*model.graph.period+len(model.layer_nodes)
        return state_position*model.graph.period >= last_tick
    return state_position >= initial_position+prefill


def _phase_summary(calls, phase, omit_first=False):
    selected = [row for row in calls if row["phase"] == phase]
    if omit_first:
        selected = selected[1:]
    elapsed = sum(row["seconds"] for row in selected)
    positions = sum(row["input_positions"] for row in selected)
    return dict(calls=len(selected), seconds=elapsed, input_positions=positions,
                positions_per_second=positions/elapsed if elapsed else None,
                median_call_seconds=statistics.median(row["seconds"] for row in selected) if selected else None,
                p95_call_seconds=sorted(row["seconds"] for row in selected)[math.ceil(.95*len(selected))-1] if selected else None)


def add_availability(measured, diagnostic):
    """Map a separate same-schedule support witness onto normal ready boundaries."""
    flags = diagnostic["output_available_by_call"]
    if len(flags) != len(measured["calls"]):
        raise ValueError("availability witness schedule differs from timed request")
    measured["first_any_backbone_output_seconds"] = next((row["ready_seconds"]
        for row, available in zip(measured["calls"], flags) if available), None)
    measured["availability_source"] = "separate full-schedule diagnostic at identical parameters/input/precision/execution policy; not a timed support capture"


def run_request(model, x, schedule, *, precision="fp32", observer=None, timing=True):
    """One zero-state request; retain only bounded private state and scalar records.

    ``observer`` is exclusively for a separate untimed diagnostic. Normal runs
    include readout, all participating device fences, Python scheduling and
    output release. No output list, concatenation or host transfer is hidden.
    """
    if observer is not None and timing:
        raise ValueError("diagnostic observers must run outside normal timing")
    if x.ndim != 3 or x.shape[1] != schedule.positions or x.shape[0] < 1:
        raise ValueError("input must exactly fit the finite schedule and positive batch")
    devices = execution_devices(model, x)
    calls, state, prompt_ready_seconds = [], None, None
    synchronize_all(devices)
    start_time = time.perf_counter() if timing else 0.0
    previous_end = start_time
    output_slots = 0
    with torch.no_grad(), precision_context(x.device, precision):
        for phase, start, stop in [*schedule.calls(), ("drain", schedule.positions, schedule.positions)]:
            if phase == "drain" and (not isinstance(model, TideModel) or model.settle):
                continue
            before = time.perf_counter() if timing else 0.0
            old_position = 0 if state is None else state.position
            if phase == "drain":
                raw = model.flush(state)
                value, state = raw.readout(), raw.state
            else:
                value, state, raw = forward_state(model, x[:, start:stop], state)
            slots = value.shape[0]*value.shape[1]
            output_slots += slots
            if observer is not None:
                observer(value, raw, phase, state)
            synchronize_all(devices)
            # Release both public readout and detailed Tide return before the
            # ready timestamp. The caller does not retain a growing stream.
            del value, raw
            end = time.perf_counter() if timing else 0.0
            row = dict(phase=phase, start=start, stop=stop, input_positions=x.shape[0]*(stop-start),
                       state_before=old_position, state_after=state.position, output_slots=slots,
                       seconds=end-previous_end, forward_readout_sync_seconds=end-before,
                       ready_seconds=end-start_time)
            calls.append(row)
            previous_end = end
            if prompt_ready_seconds is None and _prompt_ready(model, state.position, schedule.prefill):
                prompt_ready_seconds = end-start_time
    synchronize_all(devices)
    end_time = time.perf_counter() if timing else 0.0
    total = end_time-start_time
    result = dict(calls=calls, complete_seconds=total, input_positions=x.shape[0]*schedule.positions,
                  complete_positions_per_second=x.shape[0]*schedule.positions/total if total else None,
                  output_slots=output_slots, final_state_position=state.position,
                  prompt_ingested_seconds=next(row["ready_seconds"] for row in calls if row["stop"] == schedule.prefill),
                  prompt_output_ready_seconds=prompt_ready_seconds,
                  prompt_output_ready_definition="first completed call whose returned logical window covers last prompt injection plus full graph-depth delay; conservative terminal-path boundary, not memory causal attribution or language-model TTFT",
                  first_incremental_call_seconds=next((row["seconds"] for row in calls if row["phase"] == "decode"), None),
                  first_any_backbone_output_seconds=None,
                  prefill=_phase_summary(calls, "prefill"),
                  decode=_phase_summary(calls, "decode"),
                  stable_decode=_phase_summary(calls, "decode", omit_first=True),
                  stable_decode_rule="omit first incremental call; no stable sample when fewer than two decode calls",
                  chunked_stream=_phase_summary(calls, "stream"),
                  drain_seconds=sum(row["seconds"] for row in calls if row["phase"] == "drain"),
                  input_window_seconds=next(row["ready_seconds"] for row in reversed(calls) if row["phase"] != "drain"),
                  state_memory=state_memory(state), synchronized_devices=[str(d) for d in devices],
                  diagnostic_checks_in_timing=False,
                  consumer="materialize backbone readout; synchronize every owner; release each output; no concatenation, task head, checksum reduction or D2H return",
                  timing_boundary="resident-input first call through final drain/readout/output release/context exit and all-owner synchronization; phase intervals include Python scheduling since prior ready boundary")
    return state, result


class OutputDiagnostic:
    """Consume one output at a time, retaining only finite/support/time records."""
    def __init__(self):
        self.available = []
        self.finite = True
        self._supports, self._times = hashlib.sha256(), hashlib.sha256()
        self.time_bounds = []
        self.readout_dtypes, self.value_dtypes, self.strength_dtypes, self.devices = set(), set(), set(), set()

    def __call__(self, value, raw, phase, state):
        self.readout_dtypes.add(str(value.dtype)); self.devices.add(str(value.device))
        self.finite &= bool(torch.isfinite(value).all())
        if raw is None:
            self.available.append(value.shape[1] > 0)
            self.time_bounds.append(dict(phase=phase, input_position_end=state.position))
        else:
            self.value_dtypes.add(str(raw.values.dtype)); self.strength_dtypes.add(str(raw.strengths.dtype))
            self.finite &= bool(torch.isfinite(raw.values).all()) and bool(torch.isfinite(raw.strengths).all())
            support = (raw.strengths > 0).detach().cpu()
            times = raw.times.detach().cpu()
            self.available.append(bool(support.any()))
            self._supports.update(str(tuple(support.shape)).encode())
            self._supports.update(support.numpy().tobytes())
            self._times.update(times.numpy().tobytes())
            self.time_bounds.append(dict(phase=phase, first_original_time=int(times[0]), last_original_time=int(times[-1]),
                                         returned_time_slots=times.numel()))

    def report(self, state):
        finite_state = all(not t.is_floating_point() or bool(torch.isfinite(t).all()) for t in _tensors(state))
        return dict(finite_outputs=self.finite, finite_private_state=finite_state,
                    output_available_by_call=self.available, output_support_sha256=self._supports.hexdigest(),
                    original_output_times_sha256=self._times.hexdigest(), original_time_bounds=self.time_bounds,
                    state_memory=state_memory(state), final_state_position=state.position,
                    private_state_detached=all(not t.requires_grad for t in _tensors(state)),
                    readout_dtypes=sorted(self.readout_dtypes), returned_value_dtypes=sorted(self.value_dtypes),
                    returned_strength_dtypes=sorted(self.strength_dtypes), output_devices=sorted(self.devices),
                    retention="one output window at a time; only hashes/scalars retained; separate from normal timing")


def diagnose_request(model, x, schedule, *, precision="fp32", inventory=None, audit=True):
    observer = OutputDiagnostic()
    context = StreamingWorkAudit(model, inventory=inventory) if audit and isinstance(model, TideModel) else nullcontext()
    with context as observed:
        state, _ = run_request(model, x, schedule, precision=precision, observer=observer, timing=False)
    report = observer.report(state)
    report["execution"] = dict(requested_region_concurrency=getattr(model, "_placement_spec", {}).get("region_concurrency", "serial"),
                               effective_region_concurrency=getattr(model, "_placement_spec", {}).get("region_concurrency", "serial"),
                               hooks="ordinary Python method/module hooks with locked counters; no thread-local dispatch mode")
    return report, observed.report() if observed is not None else None


def bounded_parameter_identity(model, chunk_bytes=8 << 20):
    """Same parameter hash convention as workloads.py, with bounded host copies."""
    digest, finite = hashlib.sha256(), True
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            digest.update(name.encode()); digest.update(str(tuple(parameter.shape)).encode())
            if not parameter.is_contiguous():
                raise ValueError("benchmark identity requires contiguous parameter storage")
            flat = parameter.detach().view(-1)
            elements = max(1, chunk_bytes//parameter.element_size())
            for start in range(0, flat.numel(), elements):
                piece = flat[start:start+elements].cpu()
                digest.update(piece.view(torch.uint8).numpy().tobytes())
                finite &= bool(torch.isfinite(piece).all())
    return dict(sha256=digest.hexdigest(), finite=finite, host_copy_chunk_bytes=chunk_bytes,
                scope="dense initial named Parameters before ownership placement; shape/name/raw contiguous bytes")


def _canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_identity():
    import tide
    package = Path(tide.__file__).resolve().parent
    harness = Path(__file__).resolve().parent
    def tree_hash(path):
        digest = hashlib.sha256()
        for file in sorted(path.rglob("*.py")):
            digest.update(str(file.relative_to(path)).encode()); digest.update(file.read_bytes())
        return digest.hexdigest()
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        head = "unavailable"
    return dict(source_commit=os.environ.get("TIDE_SOURCE_COMMIT", head),
                harness_commit=os.environ.get("TIDE_HARNESS_COMMIT", head),
                tide_content_sha256=tree_hash(package), harness_content_sha256=tree_hash(harness),
                tide_path=str(package), harness_path=str(harness))


def make_reference(model, config, schedule, *, batch, seed, precision, identity, parameter_identity,
                   input_sha256, diagnostic, audit, placement):
    if audit is None:
        raise ValueError("formal matching requires a full actual workload audit")
    if not parameter_identity["finite"] or not diagnostic["finite_outputs"] or not diagnostic["finite_private_state"]:
        raise ArithmeticError("nonfinite Tide reference cannot define formal comparisons")
    payload = dict(schema="tide-inference-reference-1", config=config.to_dict(), schedule=asdict(schedule),
                   batch=batch, seed=seed, precision=precision, mode="inference", source=identity,
                   tide_parameters=parameter_identity, tide_input_sha256=input_sha256,
                   tide_input_width=config.input_dim or config.d, placement=placement,
                   diagnostics=diagnostic, audit=audit,
                   matches=match_transformers(len(model.graph.nodes), len(model.graph.widths),
                                              audit["parameters"]["unique_parameters"], audit["effective_touched"]["mean"]),
                   matching_boundary="zero initial state; actual prefill/decode/chunked schedule and sole final Timed drain; frozen constructors for every later timed sample",
                   comparisons_equivalent=False)
    return dict(payload, reference_sha256=_canonical_hash(payload))


def validate_reference(reference, config, schedule, *, batch, seed, precision, identity):
    if reference.get("schema") != "tide-inference-reference-1":
        raise ValueError("unknown inference reference schema")
    payload = {key: value for key, value in reference.items() if key != "reference_sha256"}
    if _canonical_hash(payload) != reference.get("reference_sha256"):
        raise ValueError("inference reference digest mismatch")
    expected = dict(config=config.to_dict(), schedule=asdict(schedule), batch=batch, seed=seed, precision=precision, mode="inference")
    for key, value in expected.items():
        if reference.get(key) != value:
            raise ValueError(f"inference reference {key} differs from requested workload")
    if reference["source"]["tide_content_sha256"] != identity["tide_content_sha256"]:
        raise ValueError("inference reference Tide source differs from current implementation")
    if not reference["diagnostics"]["finite_outputs"] or not reference["diagnostics"]["finite_private_state"]:
        raise ArithmeticError("inference reference is nonfinite")
    return reference


def allocator_stats(devices, *, reset=False):
    result = {}
    for device in dict.fromkeys(torch.device(d) for d in devices):
        if device.type == "cpu":
            continue
        backend = getattr(torch, device.type)
        if reset:
            backend.reset_peak_memory_stats(device)
        result[str(device)] = dict(allocated_bytes=backend.memory_allocated(device), reserved_bytes=backend.memory_reserved(device),
                                   peak_allocated_bytes=backend.max_memory_allocated(device), peak_reserved_bytes=backend.max_memory_reserved(device))
    return result


def _persist(path, record):
    from scripts.job import atomic
    atomic(path, record)


def _placement(model, args, devices):
    if isinstance(model, TideModel):
        if args.placement == "single":
            model.to(devices[0])
            return model, dict(kind="single", devices=[str(devices[0])])
        from tide.placement import place_model, placement_report
        from tide.placement_ownership import balanced_region_partitions
        kwargs = dict(transport=args.transport)
        if args.placement == "region":
            kwargs.update(region_partitions=balanced_region_partitions(model, len(devices)),
                          region_concurrency=args.region_concurrency)
        model = place_model(model, devices, **kwargs)
        return model, placement_report(model)
    # Each Transformer block is an indivisible owner. C/D may contain fewer
    # blocks than admitted devices; report the nonparticipating devices openly.
    used = devices[:min(len(devices), len(model.blocks))]
    if len(used) == 1:
        model.to(used[0])
        return model, dict(kind="single", devices=[str(used[0])], unused_admitted_devices=[str(d) for d in devices[1:]])
    model = LayerPlacedTransformer(model, used, transport=args.transport)
    return model, dict(model.placement_report(), unused_admitted_devices=[str(d) for d in devices[len(used):]])


def _check_budget(memory, args):
    for device, values in memory.items():
        for kind in ("allocated", "reserved"):
            limit = getattr(args, f"{kind}_budget_gib")
            if limit is not None and values[f"peak_{kind}_bytes"] > limit*2**30:
                raise MemoryError(f"{device} observed {kind} peak exceeds {limit} GiB sampled budget")


def _clear_model_cache(devices):
    gc.collect()
    for device in dict.fromkeys(devices):
        if device.type != "cpu":
            backend = getattr(torch, device.type)
            with backend.device(device):
                backend.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-overrides", type=json.loads, default={},
                        help="JSON top-level config overrides, included in the resolved record/reference")
    parser.add_argument("--subjects", nargs="+", choices=("tide", "A", "B", "C", "D"), default=["tide"])
    parser.add_argument("--reference", type=Path, help="frozen reference JSON, or an earlier successful inference result")
    parser.add_argument("--reference-output", type=Path, help="write the small Tide matching reference separately")
    parser.add_argument("--audit", choices=("full", "none"), default="full")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--devices", default="cpu", help="comma-separated admitted logical devices")
    parser.add_argument("--placement", choices=("single", "layer", "region"), default="single", help="Tide owner policy; Transformer uses contiguous layers")
    parser.add_argument("--transport", choices=("auto", "direct", "host_stage", "synchronized"), default="auto")
    parser.add_argument("--region-concurrency", choices=("serial", "per_device"), default="serial")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--prefill", type=int, default=128)
    parser.add_argument("--prefill-chunk", type=int, default=128)
    parser.add_argument("--decode", type=int, default=256)
    parser.add_argument("--stream-positions", type=int, default=0)
    parser.add_argument("--stream-chunk", type=int, default=128)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--cold", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--transformer-chunk", type=int, default=128)
    parser.add_argument("--transformer-backend", choices=("auto", "math", "tiled", "sdpa", "npu_fusion"), default="auto")
    parser.add_argument("--transformer-contiguous-qkv", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=751)
    parser.add_argument("--allocated-budget-gib", type=float)
    parser.add_argument("--reserved-budget-gib", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not isinstance(args.config_overrides, dict):
        parser.error("config overrides must be a JSON object")
    try:
        schedule = InferenceSchedule(args.prefill, args.prefill_chunk, args.decode, args.stream_positions, args.stream_chunk)
    except ValueError as error:
        parser.error(str(error))
    if min(args.batch, args.samples, args.threads, args.transformer_chunk) < 1 or args.warmup < 0:
        parser.error("batch/samples/threads/chunk must be positive; warmup may be zero")
    if len(set(args.subjects)) != len(args.subjects) or "tide" in args.subjects and args.subjects[0] != "tide":
        parser.error("subjects must be unique; Tide must precede its frozen comparisons")
    if any(subject != "tide" for subject in args.subjects) and args.reference is None and "tide" not in args.subjects:
        parser.error("A-D require --reference or an initial Tide subject")
    if args.audit == "none" and (args.subjects != ["tide"] or args.reference_output or args.audit_only):
        parser.error("--audit none is only a Tide performance screen without formal matching")
    if args.audit_only and args.subjects != ["tide"]:
        parser.error("--audit-only requires just the Tide subject")
    if args.reference_output and "tide" not in args.subjects:
        parser.error("--reference-output requires a Tide subject")
    if args.reference and "tide" in args.subjects:
        parser.error("a new Tide subject defines its own reference; omit --reference")
    if args.region_concurrency != "serial" and args.placement != "region":
        parser.error("per_device concurrency requires Tide region placement")
    for path in (args.output, args.reference_output):
        if path is not None and path.exists():
            parser.error("choose new output/reference identities")
    if args.reference_output == args.output:
        parser.error("output and reference-output must be distinct")
    for value in (args.allocated_budget_gib, args.reserved_budget_gib):
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error("sampled device budgets must be finite and positive")
    device_names = args.devices.split(",")
    if any(name.startswith("npu") for name in device_names):
        import torch_npu
        torch.npu.set_device(device_names[0])
        torch.npu.matmul.allow_hf32 = False
        torch.npu.conv.allow_hf32 = False
    devices = tuple(torch.device(name) for name in device_names)
    if not devices or len({d.type for d in devices}) != 1 or devices[0].type not in {"cpu", "cuda", "npu"}:
        parser.error("devices must use one CPU/CUDA/NPU backend")
    if args.placement == "single" and len(devices) > 1 and "tide" in args.subjects:
        parser.error("choose layer/region placement for multiple Tide devices")
    torch.set_num_threads(args.threads)
    config = ModelConfig.from_dict(json.loads(args.config.read_text()) | args.config_overrides)
    identity = source_identity()
    record = dict(schema="tide-inference-results-1", source=identity,
                  arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  config=config.to_dict(), schedule=asdict(schedule), precision=precision_policy(devices[0], args.precision),
                  resource_identity=os.environ.get("TIDE_RESOURCE_IDENTITY", "unspecified"),
                  physical_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"), torch=str(torch.__version__),
                  cpu_affinity=sorted(os.sched_getaffinity(0)), torch_threads=torch.get_num_threads(), host_load_before=os.getloadavg(),
                  model_lifetime="construct/hash/place one subject; release all state/weights before constructing next; no simultaneous dense Tide and Transformer",
                  request_boundary="zero state for cold, diagnostic, every warmup and every sample; real continued private state inside each finite request; only one final Timed drain",
                  allocator_boundary="discard diagnostic free blocks before independent warmup; keep warmup cache for normal samples; reset peaks separately per cold/sample",
                  cold_boundary="first invocation of this subject before its workload audit/warmup; preceding subjects can warm process/kernel caches; fresh-process comparisons require separate one-subject launches; excludes import/construction/hash/initial weight and input transfer",
                  state="running", last_stage="initialization", subjects={})
    _persist(args.output, record)
    current = None
    try:
        reference = None
        if args.reference is not None:
            loaded = json.loads(args.reference.read_text())
            reference = loaded.get("matching_reference", loaded)
            validate_reference(reference, config, schedule, batch=args.batch, seed=args.seed, precision=args.precision, identity=identity)
            record["reference_file_sha256"] = hashlib.sha256(args.reference.read_bytes()).hexdigest()
            record["matching_reference"] = reference
        for subject in args.subjects:
            current = record["subjects"][subject] = dict(state="running", observations=[])
            record["last_stage"] = subject+":construct"
            _persist(args.output, record)
            torch.manual_seed(args.seed)
            if subject == "tide":
                model = TideModel(config)
                inventory = FrozenParameterInventory(model) if args.audit == "full" else None
                width = config.input_dim or config.d
                current["comparison"] = dict(kind="tide", mathematical_equivalence_to_transformer=False)
            else:
                selected = reference["matches"][subject]
                model = TransformerBaseline(**selected["constructor"], chunk=args.transformer_chunk,
                                            backend=args.transformer_backend, contiguous_qkv=args.transformer_contiguous_qkv)
                inventory, width = None, model.d
                current["comparison"] = dict(kind="joint", joint_class=subject, match=selected,
                                              reference_sha256=reference["reference_sha256"], comparisons_equivalent=False)
            current["initial_parameters"] = bounded_parameter_identity(model)
            if not current["initial_parameters"]["finite"]:
                raise ArithmeticError("initial parameters are nonfinite")
            current["parameters"] = parameter_count(model)
            if subject != "tide" and current["parameters"] != selected["actual"]["parameters"]:
                raise AssertionError("actual Transformer parameters differ from frozen joint match")
            model, placement = _placement(model, args, devices)
            model.eval()
            current["placement"] = placement
            current["spec"] = config.to_dict() if subject == "tide" else model.configuration()
            x_cpu = paired_input(args.batch, schedule.positions, width, args.seed+1)
            current["input_sha256"], current["input_shape"] = tensor_hash(x_cpu), list(x_cpu.shape)
            prefix = min(width, config.input_dim or config.d)
            current["shared_input_prefix"] = dict(width=prefix, sha256=tensor_hash(x_cpu[..., :prefix]),
                rule="feature-wise seeded CPU generation, same sample/position/feature prefix at different widths")
            x = x_cpu.to(devices[0]); del x_cpu
            active_devices = execution_devices(model, x)
            synchronize_all(active_devices)
            _persist(args.output, record)
            if args.cold and not args.audit_only:
                record["last_stage"] = subject+":cold"
                _persist(args.output, record)
                allocator_stats(active_devices, reset=True)
                state, cold = run_request(model, x, schedule, precision=args.precision)
                cold["allocator"] = allocator_stats(active_devices)
                del state
                current["cold"] = cold
                cold["preceding_subjects_in_process"] = list(record["subjects"])[:-1]
                _persist(args.output, record)
                _check_budget(cold["allocator"], args)
            record["last_stage"] = subject+":diagnostic"
            _persist(args.output, record)
            diagnostic, audit = diagnose_request(model, x, schedule, precision=args.precision, inventory=inventory, audit=args.audit == "full")
            current["diagnostics"] = diagnostic
            if not diagnostic["finite_outputs"] or not diagnostic["finite_private_state"]:
                raise ArithmeticError("nonfinite inference diagnostic")
            if subject == "tide" and audit is not None:
                reference = make_reference(model, config, schedule, batch=args.batch, seed=args.seed, precision=args.precision,
                    identity=identity, parameter_identity=current["initial_parameters"], input_sha256=current["input_sha256"],
                    diagnostic=diagnostic, audit=audit, placement=placement)
                record["matching_reference"] = reference
                if args.reference_output:
                    _persist(args.reference_output, reference)
            if "cold" in current:
                add_availability(current["cold"], diagnostic)
            _persist(args.output, record)
            if not args.audit_only:
                # Intrusive support/finite reductions can reserve buffers that
                # normal inference never needs. Start warmup with only live
                # weights/input retained, then preserve its normal reuse pool.
                _clear_model_cache(active_devices)
                record["last_stage"] = subject+":warmup"
                _persist(args.output, record)
                for _ in range(args.warmup):
                    state, _ = run_request(model, x, schedule, precision=args.precision)
                    del state
                for index in range(args.samples):
                    record["last_stage"] = f"{subject}:sample:{index}"
                    _persist(args.output, record)
                    allocator_stats(active_devices, reset=True)
                    state, measured = run_request(model, x, schedule, precision=args.precision)
                    measured["allocator"] = allocator_stats(active_devices)
                    del state
                    add_availability(measured, diagnostic)
                    current["observations"].append(measured)
                    _persist(args.output, record)
                    _check_budget(measured["allocator"], args)
            current["state"] = "completed"
            current["process_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            _persist(args.output, record)
            del model, inventory, x
            _clear_model_cache(active_devices)
        record.update(state="completed", last_stage="complete", host_load_after=os.getloadavg())
        _persist(args.output, record)
        print(json.dumps(dict(state=record["state"], output=str(args.output), subjects=args.subjects)))
        return 0
    except Exception as error:
        if current is not None:
            current.update(state="failed", error=f"{type(error).__name__}: {error}")
        record.update(state="failed", error=f"{type(error).__name__}: {error}", process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        _persist(args.output, record)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
