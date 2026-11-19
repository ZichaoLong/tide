"""Finite qualification of each memory binding at its own fixed parameters.

Different mathematical candidates never compare outputs. Each candidate compares
ordinary and batched/recomputed execution, then performs real continuing AdamW
updates and restores in a new interpreter. Timings are intrusive diagnostics.
"""
from dataclasses import asdict, replace
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess

import torch

from tide import ClockConfig, ModelConfig, TideModel, TopologyConfig
from tide.precision import ArithmeticAudit, precision_context, precision_policy
from benchmarks.cache_metadata import CacheMetadataDiagnostic
from benchmarks.inference import InferenceSchedule, run_request
from benchmarks.workloads import paired_input
from scripts.check_continuation import witness as continuation_witness
from scripts.check_training_paths import package_identity, run as training_paths
from scripts.job import atomic
from scripts.memory_candidate_formulas import formula_witness
from scripts.qualify_precision import THRESHOLDS, tensors


CANDIDATES = ("x", "all_ssd", "reverse", "kv_half", "kv_third", "kv_sixth", "memory_shared")
HARD_FAILURES = {"nonfinite", "metadata_failed", "support_failed", "execution_failed", "structural_failed", "exact_contract_failed"}


def candidate_configs(base=None):
    base = base or ModelConfig(topology=TopologyConfig(kind="multiscale", profile="compact-5-dd4", n=2, hubs=1),
                               d=192, memory="X", clock=ClockConfig(kv_window=16), attention_chunk=4)
    if base.sharing != "independent":
        raise ValueError("this matrix isolates memory binding with independent non-memory parameters")
    if base.d % 192:
        raise ValueError("this seven-point matrix requires d divisible by 192 for all three 32-channel widths")
    common = dict(memory="X", memory_sharing=None, kv_width=None)
    base = replace(base, **common)
    return {"x": base, "all_ssd": replace(base, memory="S"),
            "reverse": replace(base, memory="X_reverse"),
            "kv_half": replace(base, kv_width=base.d // 2),
            "kv_third": replace(base, kv_width=base.d // 3),
            "kv_sixth": replace(base, kv_width=base.d // 6),
            "memory_shared": replace(base, memory_sharing="region_shared", adapter_rank=8)}


def aggregate(states):
    unknown = set(states) - (HARD_FAILURES | {"within_threshold", "finite_deviation"})
    if unknown:
        raise ValueError("unknown qualification states: " + repr(sorted(unknown)))
    return next((name for name in ("execution_failed", "nonfinite", "exact_contract_failed", "structural_failed",
                                  "metadata_failed", "support_failed", "finite_deviation") if name in states), "within_threshold")


def comparison_summary(record):
    """Do not hide discrete or dtype failures behind an inherited finite label."""
    counts = dict(comparisons=0, nonfinite=0, metadata=0, dtype=0, support=0, tolerance=0)
    fields = dict(nonfinite=[], metadata=[], dtype=[], support=[], tolerance=[])
    names = dict(nonfinite="nonfinite", metadata="metadata_failures", dtype="dtype_differences",
                 support="support_differences", tolerance="threshold_failures")
    def visit(value, path="root"):
        if isinstance(value, dict):
            if all(key in value for key in names.values()):
                counts["comparisons"] += 1
                for category, key in names.items():
                    counts[category] += len(value[key])
                    if value[key] and len(fields[category]) < 12:
                        fields[category].append(path)
            else:
                for key, part in value.items():
                    visit(part, path + "." + str(key))
        elif isinstance(value, (tuple, list)):
            for index, part in enumerate(value):
                visit(part, path + "." + str(index))
    visit(record)
    state = ("nonfinite" if counts["nonfinite"] else "structural_failed" if counts["dtype"] else
             "metadata_failed" if counts["metadata"] else "support_failed" if counts["support"] else
             "finite_deviation" if counts["tolerance"] else "within_threshold")
    return dict(state=state, counts=counts, first_comparison_paths=fields,
                support_policy="positive support differs: separate discrete failure, never only a finite-tolerance label")


def arithmetic_witness(config, device, precision):
    """Separate intrusive dispatch witness; never included in normal timing."""
    torch.manual_seed(2420)
    model = TideModel(config).to(device)
    value = paired_input(2, 4, config.input_dim or config.d, 2421).to(device).requires_grad_()
    with precision_context(device, precision), ArithmeticAudit() as audit:
        result = model(value, record_work=True)
        loss = result.readout().square().mean()
    loss.backward()
    operators = audit.report()
    nonfinite = [path for path, tensor in tensors(result) if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all())]
    gradients = {name: (None if parameter.grad is None else int(torch.count_nonzero(parameter.grad)))
                 for name, parameter in model.named_parameters()}
    nonfinite += ["gradient."+name for name, parameter in model.named_parameters()
                  if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())]
    if value.grad is not None and not bool(torch.isfinite(value.grad).all()):
        nonfinite.append("input_gradient")
    input_nonzero = None if value.grad is None else int(torch.count_nonzero(value.grad))
    memory_nonzero = any(count for name, count in gradients.items() if name.startswith("memories.") and ".ffn." not in name)
    connected = bool(input_nonzero and memory_nonzero)
    bf16 = any("bfloat16" in dtype for row in operators for dtype in row["outputs"])
    return dict(state="nonfinite" if nonfinite else "structural_failed" if not connected or precision == "bf16" and not bf16 else "within_threshold",
                nonfinite=nonfinite, bf16_matrix_observed=bf16, operators=operators,
                input_gradient_nonzeros=input_nonzero, parameter_gradient_nonzeros=gradients,
                nonzero_input_and_memory_gradient=connected,
                parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
                persistent_dtypes=sorted({str(t.dtype) for _, t in tensors(result.state)}),
                actual_work=result.work, scope="separate B2/L4 dispatch-visible forward dtype and complete VJP observation; not hardware accumulator proof")


def cache_metadata_witness(config, device, precision):
    """Observe actual positive writes and evictions without changing the clock."""
    torch.manual_seed(2422)
    model = TideModel(config).to(device).eval()
    schedule = InferenceSchedule(prefill=4, prefill_chunk=2, decode_steps=5)
    value = paired_input(1, schedule.positions, config.input_dim or config.d, 2423).to(device)
    possible = ("K" in model.memories and (schedule.positions-1)*model.graph.period >= model.memories["K"].window)
    observer = CacheMetadataDiagnostic(model, schedule, 1, sample_every=1)
    with observer.ledger:
        state, _ = run_request(model, value, schedule, precision=precision, observer=observer, timing=False)
    diagnosis = observer.report(state, require_kv_eviction=possible)
    return dict(state="within_threshold" if diagnosis["passed"] else "metadata_failed",
                scope="independent positive-write ledger and retained original times; B1/prefill4/chunk2/decode5, separate untimed call",
                schedule=asdict(schedule),
                kv_eviction_required=possible,
                eviction_scope="positive eviction required where the finite clock span reaches the KV window; a larger configured window remains explicitly uncovered",
                diagnosis=diagnosis)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--config", type=Path, help="optional common ModelConfig JSON; d divisible by 192")
    parser.add_argument("--candidates", choices=CANDIDATES, nargs="+", default=list(CANDIDATES))
    parser.add_argument("--paths", choices=("baseline", "batch4", "combined"), nargs="+", default=["baseline", "combined"])
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--length", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--skip-formulas", action="store_true", help="reuse separately identified FP32 primitive evidence")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    directory = args.output.with_suffix(".data")
    if args.output.exists() or directory.exists():
        parser.error("choose a new output and sibling .data identity")
    if not 1 <= args.steps <= 20 or min(args.batch, args.length, args.chunk, args.threads) < 1:
        parser.error("steps must be 1..20; batch, length, chunk and threads must be positive")
    if args.paths[0] != "baseline" or len(set(args.paths)) != len(args.paths) or len(set(args.candidates)) != len(args.candidates):
        parser.error("paths must start with baseline and both lists must be unique")
    if args.device.startswith("npu"):
        import torch_npu
        torch.npu.set_device(args.device)
        torch.npu.matmul.allow_hf32 = False
        torch.npu.conv.allow_hf32 = False
    torch.set_num_threads(args.threads)
    base = ModelConfig.from_dict(json.loads(args.config.read_text())) if args.config else None
    configs = candidate_configs(base)
    directory.mkdir(parents=True)
    report = dict(schema="tide-memory-candidates-v1", state="running",
                  source=os.environ.get("TIDE_SOURCE_COMMIT") or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  package_identity=package_identity(), harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  torch=str(torch.__version__), precision=precision_policy(args.device, args.precision),
                  physical_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                  cpu_affinity=sorted(os.sched_getaffinity(0)), thresholds=THRESHOLDS[args.precision],
                  comparison_scope="same mathematical candidate and fixed parameters; no cross-candidate equivalence",
                  formula_precision="fp32, even when integrated matrix arithmetic is BF16",
                  timing_scope="intrusive correctness only; not normal inference or training throughput", formulas=[], rows=[])
    atomic(args.output, report)
    if not args.skip_formulas:
        for width in (96, 64, 32):
            try:
                report["formulas"].append(formula_witness(width, device=args.device))
                report["formulas"][-1]["comparison_summary"] = comparison_summary(report["formulas"][-1]["checks"])
                report["formulas"][-1]["state"] = aggregate([report["formulas"][-1]["state"], report["formulas"][-1]["comparison_summary"]["state"]])
            except Exception as error:
                import traceback
                traceback.print_exc()
                report["formulas"].append(dict(width=width, state="execution_failed", error=repr(error)))
            atomic(args.output, report)
    diagnosed_continuation = False
    for name in args.candidates:
        config = configs[name]
        row = dict(candidate=name, config=config.to_dict(), state="running")
        report["rows"].append(row)
        atomic(args.output, report)
        try:
            row["arithmetic"] = arithmetic_witness(config, args.device, args.precision)
            row["cache_metadata"] = cache_metadata_witness(config, args.device, args.precision)
            # This schedule checks continuation; the separate ledger establishes
            # whether actual writes reached eviction for the configured window.
            row["whole_chunk"] = continuation_witness(config, (3, 2, 4), device=args.device, precision=args.precision)
            if (not diagnosed_continuation and row["whole_chunk"]["state"] == "finite_deviation"
                    and config.d <= 192 and config.topology.n <= 2):
                row["bounded_cpu_fp64"] = continuation_witness(config, (3, 2, 4), device="cpu", precision="fp64")
                row["bounded_cpu_fp64"]["interpretation"] = "same seed and whole/chunk question in CPU FP64; does not establish the cause of the original device/precision deviation"
                diagnosed_continuation = True
            case_dir = directory/name
            case_dir.mkdir()
            def progress(record):
                row["paths"] = record
                atomic(args.output, report)
            row["paths"] = training_paths(config, device=args.device, precision=args.precision,
                candidates=args.paths, batch=args.batch, length=args.length, steps=args.steps, chunk=args.chunk,
                directory=case_dir, new_process=True, witness=True, progress=progress)
            row["comparison_summary"] = comparison_summary(dict(whole_chunk=row["whole_chunk"], paths=row["paths"]))
            row["state"] = aggregate([row["whole_chunk"]["state"], row["paths"]["state"],
                                      row["comparison_summary"]["state"], row["arithmetic"]["state"],
                                      row["cache_metadata"]["state"]])
        except Exception as error:
            import traceback
            traceback.print_exc()
            row.update(state="execution_failed", error=repr(error))
        atomic(args.output, report)
        gc.collect()
        if args.device.startswith("npu"):
            torch.npu.empty_cache()
    report["state"] = aggregate([r["state"] for r in report["rows"] + report["formulas"]])
    atomic(args.output, report)
    print(json.dumps(dict(state=report["state"], rows=[dict(candidate=r["candidate"], state=r["state"]) for r in report["rows"]])))
    return int(report["state"] in HARD_FAILURES)


if __name__ == "__main__":
    raise SystemExit(main())
