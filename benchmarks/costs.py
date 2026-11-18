"""Finite paired complete-training costs, with separate restored diagnostics.

Normal samples have no hooks or profiler. Every diagnostic restores the exact
first measured parameter/gradient/AdamW/RNG state. Each step is fresh prefill,
required Timed flush, readout/loss, backward, AdamW, sync, scalar and release.
This is not incremental training; use benchmarks.continuous for that schedule.
"""
import argparse
from contextlib import nullcontext
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time
import traceback

import torch

from tide import ModelConfig, TideModel
from tide.precision import precision_context, precision_policy
from benchmarks.baselines import TransformerBaseline, parameter_count
from benchmarks.matching import match_transformers
from benchmarks.references import file_identity, load_reference
from benchmarks.run import _clear, _device_memory, _loss, _storage, check_memory, synchronize
from benchmarks.stages import SavedAudit, ShapeAudit, StageTimes, stage_hooks
from benchmarks.statistics import audit_model
from benchmarks.workloads import activity_fixture, paired_input, parameter_hash, tensor_hash
from scripts.job import atomic


TRANSFORMER_CANDIDATES = dict(standard={}, chunk256=dict(chunk=256), contiguous_qkv=dict(contiguous_qkv=True))


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def state_hash(value):
    """Stable nested tensor state identity, including dtype, shape and scalars."""
    digest = hashlib.sha256()
    def visit(item):
        digest.update(("mapping" if isinstance(item, dict) else type(item).__name__).encode())
        if isinstance(item, torch.Tensor):
            digest.update(str((item.dtype, tuple(item.shape))).encode())
            digest.update(item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=repr):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for part in item:
                visit(part)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def _rng(device):
    result = dict(cpu=torch.get_rng_state())
    if device.type in {"npu", "cuda"}:
        result[device.type] = getattr(torch, device.type).get_rng_state(device)
    return result


class TrainingAnchor:
    """CPU-owned exact anchor; snapshot, hashing and restore are outside timing."""
    def __init__(self, model, optimizer, device):
        self.device = device
        self.model = _cpu_copy(model.state_dict())
        self.optimizer = _cpu_copy(optimizer.state_dict())
        self.gradients = _cpu_copy([p.grad for p in model.parameters()])
        self.rng = _cpu_copy(_rng(device))
        self.identity = dict(model_sha256=state_hash(self.model), optimizer_sha256=state_hash(self.optimizer),
                             gradients_sha256=state_hash(self.gradients), rng_sha256=state_hash(self.rng))
        from torch.utils._pytree import tree_flatten
        self.cpu_tensor_bytes = sum(t.numel()*t.element_size() for t in
                                    tree_flatten((self.model, self.optimizer, self.gradients, self.rng))[0]
                                    if isinstance(t, torch.Tensor))

    def restore(self, model, optimizer):
        model.load_state_dict(self.model)
        # load_state_dict may retain a CPU tensor directly; copy protects the
        # frozen optimizer anchor from CPU AdamW's in-place updates.
        optimizer.load_state_dict(_cpu_copy(self.optimizer))
        for parameter, gradient in zip(model.parameters(), self.gradients):
            parameter.grad = None if gradient is None else gradient.to(parameter.device, copy=True)
        torch.set_rng_state(self.rng["cpu"])
        if self.device.type in {"npu", "cuda"}:
            getattr(torch, self.device.type).set_rng_state(self.rng[self.device.type], self.device)
        synchronize(self.device)
        actual = dict(model_sha256=state_hash(model.state_dict()), optimizer_sha256=state_hash(optimizer.state_dict()),
                      gradients_sha256=state_hash([p.grad for p in model.parameters()]), rng_sha256=state_hash(_rng(self.device)))
        if actual != self.identity:
            raise AssertionError("training anchor restoration changed state: "+str(actual))
        return actual


def training_step(model, inputs, optimizer, device, precision, observer=None):
    scope = (lambda name: nullcontext()) if observer is None else observer.scope
    with scope("training"):
        with scope("zero_grad"):
            optimizer.zero_grad(set_to_none=True)
        with precision_context(device, precision):
            with scope("forward"):
                result = model(inputs)
            with scope("flush"):
                tail = model.flush(result.state) if hasattr(model, "graph") and not model.settle else None
            with scope("readout_loss"):
                values = ([result.readout()] if hasattr(model, "graph") else [result])
                if tail is not None:
                    values.append(tail.readout())
                loss = _loss(values)
        with scope("backward"):
            loss.backward()
        with scope("optimizer"):
            optimizer.step()
        with scope("synchronize_scalar"):
            synchronize(device)
            scalar = float(loss.detach())
        with scope("release"):
            del result, tail, values, loss
    if not math.isfinite(scalar):
        raise ArithmeticError("non-finite training loss")
    return scalar


def compact_work(audit):
    """Keep counts/definitions, omit large per-sample/time tables only."""
    if isinstance(audit, dict):
        return {key: compact_work(value) for key, value in audit.items()
                if key not in {"by_sample_time", "work_by_sample_time"}}
    return audit


def semantic_work(model, inputs, device, precision):
    if hasattr(model, "graph"):
        with precision_context(device, precision):
            result, tail, audit = audit_model(model, inputs, flush=True)
        del result, tail
        return compact_work(audit)
    batch, length, _ = inputs.shape
    layers, heads, width = len(model.blocks), model.d//model.head_dim, model.head_dim
    tiles = []
    for start in range(0, length, model.chunk):
        stop = min(length, start+model.chunk)
        lo, hi = max(0, start-model.window+1), stop
        tiles.append(dict(query_rows=stop-start, key_rows=hi-lo))
    logical_pairs = sum(min(t+1, model.window) for t in range(length))
    rectangle_pairs = sum(t["query_rows"]*t["key_rows"] for t in tiles)
    ffn = model.spec["ffn_width"]
    return dict(input_positions=batch*length, blocks=layers, heads=heads,
                attention_calls=layers*len(tiles), tiles_per_layer=tiles,
                effective_attention_pairs=batch*heads*layers*logical_pairs,
                attention_operand_rectangle_pairs=batch*heads*layers*rectangle_pairs,
                masked_rectangle_pairs=batch*heads*layers*(rectangle_pairs-logical_pairs),
                linear_forward_multiply_adds=batch*length*layers*(4*model.d**2+3*model.d*ffn),
                attention_rectangle_multiply_adds=2*batch*heads*layers*rectangle_pairs*width,
                interpretation="fresh prefill; attention operand rectangles include masked pairs, not measured native-kernel executed FLOPs; linear multiply-adds exclude norms/RoPE/activation/final norm; backward/optimizer excluded from these formula counts")


def summarize_trace(path, limit=50):
    """Retain category labels rather than guessing which vendor track is a kernel."""
    payload = json.loads(Path(path).read_text())
    events = payload.get("traceEvents", []) if isinstance(payload, dict) else payload
    groups, categories = {}, {}
    for event in events:
        if event.get("ph") != "X" or "dur" not in event:
            continue
        category = str(event.get("cat", "unspecified"))
        key = category, str(event.get("name", "unnamed"))
        row = groups.setdefault(key, dict(category=key[0], name=key[1], calls=0, inclusive_duration_us=0.))
        row["calls"] += 1
        row["inclusive_duration_us"] += float(event["dur"])
        categories[category] = categories.get(category, 0)+1
    return dict(complete_event_count=sum(categories.values()), categories=categories,
                top_by_category={cat: sorted((r for r in groups.values() if r["category"] == cat),
                                            key=lambda r: r["inclusive_duration_us"], reverse=True)[:limit]
                                 for cat in sorted(categories)},
                interpretation="raw Chrome trace categories preserved; inclusive host events nest, device events may overlap across streams, and host/device execute concurrently; sums are not wall time or additive shares; shapes/device tracks remain in raw trace")


def trace_step(model, inputs, optimizer, args, device, trace_path):
    if trace_path.exists():
        raise FileExistsError("trace already exists; preserve its observation identity: "+str(trace_path))
    if device.type == "npu":
        import torch_npu
        profiler = torch_npu.profiler
        activities = [profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU]
    else:
        profiler = torch.profiler
        activities = [profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(profiler.ProfilerActivity.CUDA)
    with profiler.profile(activities=activities, record_shapes=True) as profile:
        loss = training_step(model, inputs, optimizer, device, args.precision)
    profile.export_chrome_trace(str(trace_path))
    if not trace_path.is_file():
        raise RuntimeError("profiler did not export the requested trace")
    return dict(loss=loss, trace=file_identity(trace_path), summary=summarize_trace(trace_path),
                boundary="separate restored complete training step, without stage/dispatch/semantic hooks; profiler overhead included")


def observe(model, inputs, optimizer, args, device, row, persist, trace_path, anchor=None):
    for _ in range(args.warmup):
        training_step(model, inputs, optimizer, device, args.precision)
        check_memory(device, args.device_budget_gib)
    if anchor is None:
        anchor = TrainingAnchor(model, optimizer, device)
    row["anchor"] = anchor.identity
    row["anchor_cpu_tensor_bytes"] = anchor.cpu_tensor_bytes
    row["normal"] = dict(state="running", start_identity=anchor.restore(model, optimizer), seconds=[], losses=[],
                         memory_scope="cumulative allocator peak since immediately before normal samples; construction and warmup excluded")
    synchronize(device)
    _device_memory(device, reset=True)
    normal = row["normal"]
    persist()
    for _ in range(args.steps):
        start = time.perf_counter()
        loss = training_step(model, inputs, optimizer, device, args.precision)
        normal["seconds"].append(time.perf_counter()-start)
        normal["losses"].append(loss)
        normal["device_memory"] = check_memory(device, args.device_budget_gib)
        persist()
    median = statistics.median(normal["seconds"])
    normal.update(state="completed_finite", median_seconds=median,
                  input_positions_per_second=args.batch*args.length/median,
                  sample_range_over_median=(max(normal["seconds"])-min(normal["seconds"]))/median,
                  final_parameter_sha256=parameter_hash(model))
    normal["actual_storage"] = dict(parameters=_storage(model.parameters()),
                                    gradients=_storage(p.grad for p in model.parameters()),
                                    optimizer=_storage(t for state in optimizer.state.values() for t in state.values()
                                                       if isinstance(t, torch.Tensor)))
    if not all(bool(torch.isfinite(p).all()) and (p.grad is None or bool(torch.isfinite(p.grad).all())) for p in model.parameters()):
        raise ArithmeticError("non-finite normal endpoint parameters or gradients")
    if "work" in args.diagnostics:
        row["normal_endpoint_work"] = semantic_work(model, inputs, device, args.precision)
    row["diagnostics"] = {}
    persist()
    for kind in args.diagnostics:
        diagnostic = dict(state="running")
        row["diagnostics"][kind] = diagnostic
        persist()
        try:
            diagnostic["start_identity"] = anchor.restore(model, optimizer)
            _device_memory(device, reset=True)
            start = time.perf_counter()
            if kind == "work":
                diagnostic["report"] = semantic_work(model, inputs, device, args.precision)
            elif kind == "trace":
                diagnostic.update(trace_step(model, inputs, optimizer, args, device, trace_path))
            else:
                observer = {"stages": StageTimes, "runtime": ShapeAudit, "saved": SavedAudit}[kind]
                observer = observer(device) if kind == "stages" else observer(model) if kind == "saved" else observer()
                context = (observer if kind == "runtime" else torch.autograd.graph.saved_tensors_hooks(observer.pack, observer.unpack)
                           if kind == "saved" else nullcontext())
                with stage_hooks(model, observer), context:
                    diagnostic["loss"] = training_step(model, inputs, optimizer, device, args.precision, observer)
                diagnostic["report"] = observer.report()
                if kind == "runtime" and args.precision == "bf16" and not diagnostic["report"]["bf16_matrix_observed"]:
                    raise AssertionError("BF16 requested but no BF16 matrix result observed")
            diagnostic.update(state="completed_finite", intrusive_elapsed_seconds=time.perf_counter()-start,
                              device_memory=check_memory(device, args.device_budget_gib))
        except Exception as error:
            diagnostic.update(state="failed", error=repr(error))
            traceback.print_exc()
        persist()
    row["state"] = "failed" if any(d["state"] == "failed" for d in row["diagnostics"].values()) else "completed_finite"
    row["process_lifetime_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return anchor


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-overrides", type=json.loads, default={},
                        help="JSON top-level config overrides, included in the resolved record")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1107)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--subjects", nargs="+", choices=("tide", "transformer_A", "transformer_B", "transformer_C", "transformer_D"),
                        default=["tide", "transformer_A", "transformer_B", "transformer_C", "transformer_D"])
    parser.add_argument("--diagnostics", nargs="*", choices=("stages", "runtime", "work", "saved", "trace"),
                        default=["stages", "runtime", "work"])
    parser.add_argument("--transformer-candidates", nargs="+", choices=tuple(TRANSFORMER_CANDIDATES), default=["standard"],
                        help="standard must come first; optional chunk256 and contiguous_qkv share its post-warmup anchor")
    parser.add_argument("--activity", choices=("natural", "sparse03", "sparse10", "high"), default="natural")
    parser.add_argument("--reference-record", help="completed exact B/L initial train audit from benchmarks.run")
    parser.add_argument("--optimizer-foreach", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device-budget-gib", type=float)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if not isinstance(args.config_overrides, dict):
        parser.error("config overrides must be a JSON object")
    if min(args.batch, args.length, args.threads, args.steps) < 1 or args.warmup < 0:
        parser.error("counts must be positive; warmup may be zero")
    if args.device_budget_gib is not None and not 0 < args.device_budget_gib < float("inf"):
        parser.error("device budget must be positive and finite")
    if len(set(args.subjects)) != len(args.subjects) or len(set(args.diagnostics)) != len(args.diagnostics):
        parser.error("subjects/diagnostics must not repeat")
    if args.transformer_candidates[0] != "standard" or len(set(args.transformer_candidates)) != len(args.transformer_candidates):
        parser.error("Transformer candidates must start with standard and must not repeat")
    output = Path(args.output).resolve()
    if output.exists():
        parser.error("output already exists; choose a new observation identity")
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.device.startswith("npu"):
        import torch_npu
        torch.npu.matmul.allow_hf32 = False
        torch.npu.conv.allow_hf32 = False
    device = torch.device(args.device)
    if device.type == "npu":
        torch.npu.set_device(device)
    torch.set_num_threads(args.threads)
    config = ModelConfig.from_dict(json.loads(Path(args.config).read_text()) | args.config_overrides)
    source = os.environ.get("TIDE_SOURCE_COMMIT") or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    record = dict(schema="tide-training-costs-v1", state="running", source=source,
                  harness_source=os.environ.get("TIDE_HARNESS_COMMIT", source), arguments=vars(args),
                  config=config.to_dict(), config_file=file_identity(args.config), precision=precision_policy(device, args.precision),
                  torch=str(torch.__version__), python=sys.version, executable=sys.executable,
                  loaded_modules=os.environ.get("LOADEDMODULES"), physical_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                  resource_identity=os.environ.get("TIDE_RESOURCE_IDENTITY", "unspecified; do not infer exclusive access"),
                  cpu_affinity=sorted(os.sched_getaffinity(0)), host_load_before=os.getloadavg(),
                  working_tree_dirty=(None if os.environ.get("TIDE_SOURCE_COMMIT") else bool(subprocess.check_output(
                      ["git", "status", "--porcelain", "--", "src/tide", "benchmarks"], text=True).strip())),
                  timing=dict(schedule="fresh prefill from zero state, required Timed flush on every step", mode="train",
                              includes=["zero_grad", "preparation/packing", "forward", "Timed flush", "readout/loss", "backward", "AdamW", "sync/scalar", "release"],
                              excludes=["construction/input transfer", "warmup", "CPU anchor snapshot/restore/hash", "audit/profiler", "checkpoint IO"],
                              normal="no diagnostic hooks/profiler; sequential complete training updates",
                              diagnostics="each starts at the exact normal first-step anchor; never used as normal throughput",
                              input_positions=args.batch*args.length),
                  comparisons_equivalent=False, numerical_equivalence_qualification=False, observations=[])
    record["baseline_strength"] = dict(
        existing="fused QKV, fused SwiGLU gate/up, tensorized batch/length/heads, cached RoPE/window masks and native unit-prior attention",
        bounded_candidates=TRANSFORMER_CANDIDATES,
        rationale=dict(chunk256="fewer attention tile calls with more masked operand work; same window/function/parameters",
                       contiguous_qkv="one complete head-major Q/K/V layout before tiles; may add copies, and sliced tiles can still need adapter copies"),
        fairness="each subject's candidates use identical initial tensors and standard's exact post-warmup parameters/gradients/AdamW/RNG; candidate warmup updates are discarded before restoring that anchor; input common prefixes paired across model widths",
        limits="two finite alternatives, no claim of globally optimal Transformer; same-function tests are CPU FP64, device numerical/trajectory qualification remains separate")
    if device.type == "npu":
        record.update(torch_npu=str(torch_npu.__version__), allow_hf32=dict(matmul=False, conv=False))
    def persist():
        atomic(output, record)
    persist()
    try:
        if args.reference_record:
            args.no_flush, args.modes = False, ["train"]
            reference = load_reference(args.reference_record, config, args, same_shape=True)
            record.update(reference_record=file_identity(args.reference_record),
                          initial_tide_audit=compact_work(reference["initial_audits"]["train"]),
                          joint_matches=reference["joint_matches_by_mode"]["train"],
                          initial_tide_parameter_sha256=reference["initial_tide_parameter_sha256"])
        else:
            torch.manual_seed(args.seed)
            initial = TideModel(config)
            activity_fixture(initial, args.activity)
            record["initial_tide_parameter_sha256"] = parameter_hash(initial)
            initial = initial.to(device).train()
            inputs = paired_input(args.batch, args.length, config.input_dim or config.d, args.seed+1).to(device)
            record["initial_tide_audit"] = semantic_work(initial, inputs, device, args.precision)
            check_memory(device, args.device_budget_gib)
            record["joint_matches"] = match_transformers(len(initial.graph.nodes), len(initial.graph.widths),
                                                         parameter_count(initial), record["initial_tide_audit"]["effective_touched"]["mean"])
            del initial, inputs
            _clear(device)
        record["matching_policy"] = "A-D from initial train-mode same-precision actual B/L work; frozen constructors, no rematching after updates"
        persist()
    except Exception as error:
        record.update(state="failed", initial_audit_error=repr(error))
        persist()
        raise
    for subject in args.subjects:
        anchor = None
        for candidate in (["standard"] if subject == "tide" else args.transformer_candidates):
            row = dict(subject=subject, candidate=candidate, state="running")
            record["observations"].append(row)
            persist()
            model = optimizer = inputs = None
            try:
                if candidate != "standard" and anchor is None:
                    raise RuntimeError("standard candidate did not produce a reusable training anchor")
                torch.manual_seed(args.seed)
                if subject == "tide":
                    model = TideModel(config)
                    activity_fixture(model, args.activity)
                    row["spec"] = dict(family="tide", config=config.to_dict())
                else:
                    match = record["joint_matches"][subject[-1]]
                    model = TransformerBaseline(**(match["constructor"] | TRANSFORMER_CANDIDATES[candidate]))
                    row["spec"] = dict(model.configuration(), joint_match=match)
                    if parameter_count(model) != match["actual"]["parameters"]:
                        raise AssertionError("actual Transformer parameter count differs from joint match")
                row.update(parameters=parameter_count(model), initial_parameter_sha256=parameter_hash(model))
                model = model.to(device).train()
                width = (config.input_dim or config.d) if subject == "tide" else model.d
                inputs_cpu = paired_input(args.batch, args.length, width, args.seed+1)
                row["input_sha256"] = tensor_hash(inputs_cpu)
                inputs = inputs_cpu.to(device)
                del inputs_cpu
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=args.optimizer_foreach)
                anchor = observe(model, inputs, optimizer, args, device, row, persist,
                                 output.with_name(output.stem+"-"+subject+"-"+candidate+"-trace.json"), anchor)
            except Exception as error:
                row.update(state="failed", error=repr(error))
                if row.get("normal", {}).get("state") == "running":
                    row["normal"].update(state="failed", error=repr(error))
                traceback.print_exc()
            finally:
                model = optimizer = inputs = None
                _clear(device)
            persist()
            print(json.dumps(dict(subject=subject, candidate=candidate, state=row["state"],
                                  normal=row.get("normal", {}).get("median_seconds"))), flush=True)
        anchor = None
    record.update(state="failed" if any(r["state"] == "failed" for r in record["observations"]) else "completed_finite",
                  host_load_after=os.getloadavg())
    persist()
    return int(record["state"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
