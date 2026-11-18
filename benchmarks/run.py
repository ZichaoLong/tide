"""Finite FP32/BF16 prefill comparison at four frozen joint matches.

Training includes zero_grad, all model preparation/packing, readout/loss,
backward, AdamW and device synchronization. Timed includes its complete flush
unless --no-flush is explicitly requested. Construction, input transfer,
untimed intrusive audit, checkpoint and initialization are outside timing.
Use benchmarks.continuous for actual incremental decode and continued state.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time

import torch
from tide import ModelConfig, TideModel
from tide.precision import precision_context, precision_policy
from benchmarks.baselines import TransformerBaseline, SSDBaseline, parameter_count
from benchmarks.matching import match_transformers, parameter_memory_budget
from benchmarks.statistics import audit_model, RuntimeAudit
from benchmarks.workloads import activity_fixture, paired_input, parameter_hash, tensor_hash
from benchmarks.references import file_identity, load_reference
from scripts.job import atomic


def synchronize(device):
    if device.type == "npu":
        torch.npu.synchronize(device)
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _device_memory(device, reset=False):
    if device.type in {"npu", "cuda"}:
        backend = getattr(torch, device.type)
        if reset:
            backend.reset_peak_memory_stats(device)
        return dict(allocated_bytes=backend.memory_allocated(device),
                    reserved_bytes=backend.memory_reserved(device),
                    peak_allocated_bytes=backend.max_memory_allocated(device),
                    peak_reserved_bytes=backend.max_memory_reserved(device))
    return None


def _clear(device):
    gc.collect()
    if device.type in {"npu", "cuda"}:
        getattr(torch, device.type).empty_cache()


def _storage(tensors):
    result = {}
    for tensor in tensors:
        if tensor is None:
            continue
        key = str(tensor.device)+":"+str(tensor.dtype)
        row = result.setdefault(key, dict(elements=0, bytes=0))
        row["elements"] += tensor.numel()
        row["bytes"] += tensor.numel()*tensor.element_size()
    return result


def _forward(model, x, is_tide, flush):
    result = model(x)
    if not is_tide:
        return [result], None
    tail = model.flush(result.state) if flush and not model.settle else None
    values = [result.readout()]
    if tail is not None:
        values.append(tail.readout())
    return values, tail


def _loss(values):
    # Include all returned outputs. Timed ticks/flush are an explicit loss
    # reduction boundary, never the throughput's input-position denominator.
    return sum(value.float().square().sum() for value in values) / sum(value.numel() for value in values)


class DeviceBudgetExceeded(RuntimeError):
    """A sampled allocator peak crossed the declared budget, not an OOM."""


def check_memory(device, budget_gib):
    memory = _device_memory(device)
    if memory is not None and budget_gib is not None and memory["peak_allocated_bytes"] > budget_gib*2**30:
        raise DeviceBudgetExceeded(f"{device}: peak allocated {memory['peak_allocated_bytes']} bytes exceeds {budget_gib} GiB")
    return memory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--transformer-chunk", type=int, default=128,
                        help="explicit matched baseline query tile size; parameter matching is unchanged")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--modes", nargs="+", choices=("inference", "grad_forward", "train"),
                        default=["inference", "grad_forward", "train"])
    parser.add_argument("--subjects", nargs="+", choices=("tide", "transformer_A", "transformer_B", "transformer_C", "transformer_D",
                        "transformer_layers", "transformer_nodes", "transformer_parameters", "ssd_layers", "ssd_parameters"),
                        default=["tide", "transformer_A", "transformer_B", "transformer_C", "transformer_D"])
    parser.add_argument("--activity", choices=("natural", "sparse03", "sparse10", "high"), default="natural")
    parser.add_argument("--no-flush", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--reference-record", help="completed initial audit at this exact B/L; avoids allocating a Tide reference in each child")
    parser.add_argument("--calibration-record", help="freeze A-D constructors from this initial calibration; actual B/L audits, when available, remain separate")
    parser.add_argument("--device-budget-gib", type=float, help="stop this child after a synchronized phase/step exceeds allocator peak budget; not a hard allocator limit")
    parser.add_argument("--optimizer-foreach", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1107)
    args = parser.parse_args()
    if min(args.threads, args.batch, args.length, args.steps, args.transformer_chunk) < 1 or args.warmup < 0:
        parser.error("counts must be positive, warmup nonnegative")
    if args.device_budget_gib is not None and not 0 < args.device_budget_gib < float("inf"):
        parser.error("device budget must be positive and finite")
    if args.audit_only and (args.reference_record or args.calibration_record):
        parser.error("audit-only must observe the actual workload, without external references")
    output = Path(args.output)
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
    cfg = ModelConfig.from_dict(json.loads(Path(args.config).read_text()))
    source = os.environ.get("TIDE_SOURCE_COMMIT") or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    record = dict(schema="tide-joint-comparison-v1", source=source,
                  harness_source=os.environ.get("TIDE_HARNESS_COMMIT", source),
                  state="running", arguments=vars(args), torch=str(torch.__version__),
                  config=cfg.to_dict(), config_file=file_identity(args.config),
                  python=sys.version, executable=sys.executable, loaded_modules=os.environ.get("LOADEDMODULES"),
                  working_tree_dirty=(None if os.environ.get("TIDE_SOURCE_COMMIT") else bool(subprocess.check_output(
                      ["git", "status", "--porcelain", "--", "src/tide", "benchmarks"], text=True).strip())),
                  physical_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                  resource_identity=os.environ.get("TIDE_RESOURCE_IDENTITY", "unspecified; do not infer exclusive access"),
                  timing=dict(kind="prefill_from_zero", includes=["preparation", "packing", "returned values", "readout", "synchronization"],
                              training_adds=["zero_grad", "scalar loss", "backward", "AdamW"],
                              excludes=["model construction", "input transfer", "untimed audit", "checkpoint"],
                              timed_flush=not args.no_flush, first_output_latency_measured=False),
                  input_positions=args.batch*args.length, comparisons_equivalent=False,
                  precision=precision_policy(device, args.precision),
                  input_pairing="CPU feature-wise seed; common sample/time/feature prefixes identical across different widths; hashes identify actual tensors",
                  matching_policy="per-mode initial fixed input/parameters/zero state and precision; grad_forward/train enable autograd without an audit update; no rematching after training",
                  observations=[])
    if device.type == "npu":
        record.update(torch_npu=str(torch_npu.__version__), allow_hf32=dict(matmul=False, conv=False))
    atomic(output, record)
    initial = x_cpu = audit_inputs = None
    try:
        record["initial_audits"], record["initial_runtime_audits"], record["joint_matches_by_mode"] = {}, {}, {}
        reference = None
        if args.reference_record or args.calibration_record:
            reference = load_reference(args.reference_record or args.calibration_record, cfg, args,
                                       same_shape=bool(args.reference_record))
            record["reference_record"] = file_identity(args.reference_record or args.calibration_record)
            record["reference_source"] = reference["source"]
            record["initial_tide_parameter_sha256"] = reference["initial_tide_parameter_sha256"]
            record["activity_fixture"] = reference["activity_fixture"]
            nodes, layers = reference["topology"]["nodes"], reference["topology"]["layers"]
            target = reference["initial_audits"][args.modes[0]]["parameters"]["unique_parameters"]
            if args.reference_record:
                for key in ("initial_audits", "initial_runtime_audits", "joint_matches_by_mode", "audit_input_sha256"):
                    record[key] = reference[key]
                record["initial_audit_scope"] = "this exact initial workload; frozen external child"
            else:
                record["initial_audit_scope"] = "not measured at this B/L; constructors refer only to the calibration workload"
        else:
            torch.manual_seed(args.seed)
            initial = TideModel(cfg)
            record["activity_fixture"] = activity_fixture(initial, args.activity)
            record["initial_tide_parameter_sha256"] = parameter_hash(initial)
            target = parameter_count(initial)
            nodes, layers = len(initial.graph.nodes), len(initial.graph.widths)
            x_cpu = paired_input(args.batch, args.length, cfg.input_dim or cfg.d, args.seed+1)
            record["audit_input_sha256"] = tensor_hash(x_cpu)
            initial = initial.to(device)
            audit_inputs = x_cpu.to(device)
            check_memory(device, args.device_budget_gib)
        for reference_mode in ([] if reference is not None else args.modes):
            initial.train(reference_mode != "inference")
            with torch.set_grad_enabled(reference_mode != "inference"), precision_context(device, args.precision):
                audit_result, audit_tail, audit = audit_model(initial, audit_inputs, flush=not args.no_flush)
            for value in [audit_result.readout()] + ([] if audit_tail is None else [audit_tail.readout()]):
                if not bool(torch.isfinite(value).all()):
                    raise ArithmeticError("non-finite initial audit output")
            del value, audit_result, audit_tail
            record["initial_audits"][reference_mode] = audit
            with torch.set_grad_enabled(reference_mode != "inference"), precision_context(device, args.precision), RuntimeAudit() as runtime:
                runtime_values, runtime_tail = _forward(initial, audit_inputs, True, not args.no_flush)
            del runtime_values, runtime_tail
            record["initial_runtime_audits"][reference_mode] = runtime.report()
            if args.precision == "bf16" and not record["initial_runtime_audits"][reference_mode]["bf16_matrix_observed"]:
                raise AssertionError("BF16 requested but no BF16 matrix output observed")
            record["joint_matches_by_mode"][reference_mode] = match_transformers(nodes, layers, target, audit["effective_touched"]["mean"])
            record.setdefault("initial_audit_device_memory", {})[reference_mode] = check_memory(device, args.device_budget_gib)
        record["topology"] = dict(nodes=nodes, layers=layers)
        if args.calibration_record:
            calibration = load_reference(args.calibration_record, cfg, args, same_shape=False)
            record["calibration_record"] = file_identity(args.calibration_record)
            record["calibration_source"] = calibration["source"]
            record["calibration_arguments"] = calibration["arguments"]
            record["calibration_audits"] = calibration["initial_audits"]
            record["joint_matches_by_mode"] = calibration["joint_matches_by_mode"]
            record["matching_policy"] = "fixed calibration model; no B/L rematching; current-shape actual Tide activity is separate and may be unmeasured"
    except Exception as error:
        record.update(state="failed", initial_audit_error=repr(error),
                      failure_kind="device_budget_exceeded" if isinstance(error, DeviceBudgetExceeded) else
                      "out_of_memory" if "out of memory" in str(error).lower() else "execution_error")
        atomic(output, record)
        raise
    finally:
        del initial, x_cpu, audit_inputs
        _clear(device)
    atomic(output, record)
    if args.audit_only:
        record["state"] = "audit_complete"
        atomic(output, record)
        return 0

    def factory(subject, mode):
        if subject == "tide":
            model = TideModel(cfg)
            activity_fixture(model, args.activity)
            return model, dict(family="tide", config=cfg.to_dict(), nodes=nodes, layers=layers)
        if subject.startswith("transformer_") and subject[-1] in "ABCD":
            match = record["joint_matches_by_mode"][mode][subject[-1]]
            model = TransformerBaseline(**dict(match["constructor"], chunk=args.transformer_chunk))
            if parameter_count(model) != match["actual"]["parameters"]:
                raise AssertionError("matching formula differs from actual unique parameters")
            return model, dict(model.configuration(), joint_match=match)
        constructor = TransformerBaseline if subject.startswith("transformer") else SSDBaseline
        count = nodes if subject.endswith("nodes") else layers
        if subject.endswith("parameters"):
            with torch.device("meta"):
                one, two = constructor(cfg.d, 1), constructor(cfg.d, 2)
            per_block = parameter_count(two)-parameter_count(one)
            constant = 2*parameter_count(one)-parameter_count(two)
            count = max(1, round((target-constant)/per_block))
        model = constructor(cfg.d, count, **(dict(chunk=args.transformer_chunk) if constructor is TransformerBaseline else {}))
        return model, dict(model.configuration(), comparison_class="auxiliary_single_axis; not A/B/C/D")

    for subject in args.subjects:
        for mode in args.modes:
            row = dict(subject=subject, mode=mode, state="running", cpu_affinity=sorted(os.sched_getaffinity(0)),
                       torch_threads=torch.get_num_threads(), host_load_before=os.getloadavg())
            record["observations"].append(row)
            atomic(output, record)
            model = optimizer = inputs = None
            try:
                torch.manual_seed(args.seed)
                model, spec = factory(subject, mode)
                row.update(spec=spec, parameters=parameter_count(model), initial_parameter_sha256=parameter_hash(model))
                row["parameter_memory_budget"] = parameter_memory_budget(row["parameters"], mode=mode, precision=args.precision)
                model = model.to(device).train(mode != "inference")
                width = (cfg.input_dim or cfg.d) if subject == "tide" else model.d
                inputs_cpu = paired_input(args.batch, args.length, width, args.seed+1)
                row["input_sha256"] = tensor_hash(inputs_cpu)
                inputs = inputs_cpu.to(device)
                del inputs_cpu
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=args.optimizer_foreach) if mode == "train" else None
                row["construction_device_memory"] = check_memory(device, args.device_budget_gib)

                def step():
                    if optimizer is not None:
                        optimizer.zero_grad(set_to_none=True)
                    with torch.set_grad_enabled(mode != "inference"), precision_context(device, args.precision):
                        values, _ = _forward(model, inputs, subject == "tide", not args.no_flush)
                        loss = _loss(values) if optimizer is not None else None
                    if optimizer is not None:
                        loss.backward()
                        optimizer.step()
                    synchronize(device)
                    return None if loss is None else float(loss.detach())

                for _ in range(args.warmup):
                    step()
                    check_memory(device, args.device_budget_gib)
                synchronize(device)
                row["warmup_device_memory"] = check_memory(device, args.device_budget_gib)
                row["device_memory_before"] = _device_memory(device, reset=True)
                times, losses, sample_memory = [], [], []
                row.update(seconds=times, losses=losses, sample_device_memory=sample_memory)
                for _ in range(args.steps):
                    start = time.perf_counter()
                    loss = step()
                    times.append(time.perf_counter()-start)
                    losses.append(loss)
                    sample_memory.append(check_memory(device, args.device_budget_gib))
                    atomic(output, record)
                memory = _device_memory(device)
                median = statistics.median(times)
                with torch.set_grad_enabled(mode != "inference"), precision_context(device, args.precision):
                    if subject == "tide":
                        result, tail, final_audit = audit_model(model, inputs, flush=not args.no_flush)
                        values = [result.readout()] + ([] if tail is None else [tail.readout()])
                        del result, tail
                        row["post_timing_audit"] = final_audit
                    else:
                        with RuntimeAudit() as runtime:
                            values, _ = _forward(model, inputs, False, False)
                        row["post_timing_runtime_audit"] = runtime.report()
                        if args.precision == "bf16" and not row["post_timing_runtime_audit"]["bf16_matrix_observed"]:
                            raise AssertionError("baseline BF16 requested but no BF16 matrix output observed")
                    if not all(bool(torch.isfinite(value).all()) for value in values):
                        raise ArithmeticError("non-finite model output")
                    del values
                    if not all(bool(torch.isfinite(p).all()) and (p.grad is None or bool(torch.isfinite(p.grad).all()))
                               for p in model.parameters()):
                        raise ArithmeticError("non-finite parameters or gradients")
                row["post_audit_device_memory"] = check_memory(device, args.device_budget_gib)
                row.update(state="completed_finite", seconds=times, median_seconds=median,
                           sample_device_memory=sample_memory,
                           sample_device_memory_scope="cumulative allocator peak since post-warmup reset; read after each timed sample",
                           input_positions_per_second=args.batch*args.length/median,
                           window_latency_ms=1000*median, losses=losses,
                           process_lifetime_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                           peak_device_bytes=None if memory is None else memory["peak_allocated_bytes"],
                           device_memory=memory, numerical_equivalence_qualification=False,
                           host_load_after=os.getloadavg())
                row["device_memory_by_device"] = {} if memory is None else {str(device): memory}
                row["sample_range_over_median"] = (max(times)-min(times))/median
                row["actual_storage"] = dict(parameters=_storage(model.parameters()),
                                             gradients=_storage(p.grad for p in model.parameters()),
                                             optimizer=_storage(value for state in optimizer.state.values() for value in state.values()
                                                                if isinstance(value, torch.Tensor)) if optimizer is not None else {})
            except Exception as error:
                row.update(state="failed", error=repr(error),
                           failure_kind="device_budget_exceeded" if isinstance(error, DeviceBudgetExceeded) else
                           "out_of_memory" if "out of memory" in str(error).lower() else "execution_error")
                import traceback
                traceback.print_exc()
            atomic(output, record)
            print(json.dumps({k: row[k] for k in ("subject", "mode", "state", "parameters", "median_seconds", "input_positions_per_second", "error") if k in row}), flush=True)
            model = optimizer = inputs = None
            _clear(device)
            if row.get("failure_kind") in {"device_budget_exceeded", "out_of_memory"}:
                record["state"] = "failed"
                atomic(output, record)
                return 1
    failed = any(row["state"] != "completed_finite" for row in record["observations"])
    record["state"] = "failed" if failed else "completed_finite"
    atomic(output, record)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
