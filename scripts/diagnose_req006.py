#!/usr/bin/env python3
"""Repeat one TL-REQ-006 direct workload from fixed Git source and retained config.

Use separate processes/output directories for measure, profile and census.
This executes a frozen downstream implementation, not the foundation executor.
It uses deterministic synthetic tokens and freshly initialized weights; progress
is explicit. It does not replay a historical learning trajectory or certify
foundation continuation. Run long cases through the bounded project job service.
"""
import argparse
from contextlib import nullcontext
import copy
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

from durable_records import write_json
from req006_material import (CASES, SOURCE, configuration, extract_source,
                             synthetic_inputs, tensor_digest, sha256)
from source_identity import source_state


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--case", choices=CASES)
    source.add_argument("--config", type=Path, help="explicit small diagnostic or additional representative")
    parser.add_argument("--downstream-repo", type=Path, required=True, help="read-only Git store containing "+SOURCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--action", choices=("measure", "profile", "census"), default="measure")
    parser.add_argument("--mode", choices=("nograd", "grad-forward", "training"), default="training")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--progress-tokens", type=int, required=True,
                        help="fixed annealing/LR progress; fresh weights still do not replay that training step")
    parser.add_argument("--input-seed", type=int, default=17)
    parser.add_argument("--microbatch", type=int)
    parser.add_argument("--global-batch", type=int)
    parser.add_argument("--length", type=int)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--prime-updates", type=int, default=0,
                        help="real synthetic updates before freezing initial model/AdamW slots; distinct from warmup")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--max-parameter-bytes", type=int, default=2**31)
    parser.add_argument("--max-trace-bytes", type=int, default=8*2**30)
    args = parser.parse_args()
    if min(args.threads, args.iterations, args.max_parameter_bytes, args.max_trace_bytes) < 1:
        parser.error("threads, iterations and byte bounds must be positive")
    if min(args.warmup, args.prime_updates, args.progress_tokens) < 0:
        parser.error("warmup, prime-updates and progress must be nonnegative")
    if any(value is not None and value < 1 for value in (args.microbatch, args.global_batch, args.length)):
        parser.error("explicit dimensions must be positive")
    if args.action != "measure" and args.iterations != 1:
        parser.error("each instrumented process records one request/update")
    if args.device.split(":")[0] not in {"cpu", "cuda", "npu"}:
        parser.error("choose an explicit device; automatic substitution is not supported")
    return args


def main():
    args = arguments()
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source, dirty = source_state(root)
    record = dict(schema="tide-req006-diagnostic-v1", state="running", source=source, dirty=dirty,
                  request={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  diagnostic_only=args.action != "measure", iterations=[],
                  correctness="not_checked; frozen downstream numerical guards remain enabled",
                  foundation_execution=False, independent_recovery="not_checked",
                  historical_trajectory_replay=False,
                  affinity=sorted(os.sched_getaffinity(0)),
                  timing_scope=dict(complete="CPU tensor partition/copy/upload, target counting, model built-in "
                      "observations, selected forward/update, disposal and final synchronization",
                      update="Frozen campaign_runtime.logical_update timer: zero_grad, forward/loss/backward "
                      "for all microbatches, observation/scalar reads, clip, AdamW, terminal sync; "
                      "excludes upload and pre-timer target counting/progress setup",
                      excluded="model/input construction, state reset, warmup, identity hashing, "
                      "profiling/export, evaluation, checkpoint and log I/O",
                      job="wall time of this diagnostic is recorded separately; not a learning-job throughput"))
    write_json(output/"result.json", record)
    started = time.perf_counter()
    try:
        config, origin = configuration(args.case, args.config)
        original = copy.deepcopy(config)
        for name, value in (("batch", args.microbatch), ("global_batch", args.global_batch), ("length", args.length)):
            if value is not None:
                config["training"][name] = value
        record.update(configuration_origin=origin, original_configuration=original, effective_configuration=copy.deepcopy(config),
                      configuration_overrides={k: config["training"][k] for k in ("batch", "global_batch", "length")
                                               if config["training"][k] != original["training"][k]})
        record["downstream_source"] = extract_source(args.downstream_repo, output/"source")
        if any(name == "tide_lab" or name.startswith("tide_lab.") for name in sys.modules):
            raise RuntimeError("a downstream package is already imported; use a fresh process")
        sys.path.insert(0, str(output/"source"))
        sys.path.insert(0, str(root/"python"))
        import torch
        from tidegraph.runtime import resolve_device, manifest
        from req006_runtime import capture, restore, run_once, reset_memory_peak, memory
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        device, reason = resolve_device(args.device)
        dtype = getattr(torch, config["dtype"])
        if dtype != torch.float32:
            raise ValueError("this frozen campaign diagnostic currently preserves its FP32 model contract")
        torch.manual_seed(config["seed"])
        runtime = importlib.import_module("tide_lab.campaign_runtime")
        if not Path(runtime.__file__).resolve().is_relative_to(output/"source"):
            raise RuntimeError("downstream runtime import escaped the fixed extraction")
        record["runtime"] = manifest(device, reason, dtype)
        record["threads"] = dict(intraop=torch.get_num_threads(), interop=torch.get_num_interop_threads())
        record["dispatch_environment"] = {key: os.environ.get(key) for key in
            ("ASCEND_RT_VISIBLE_DEVICES", "ASCEND_LAUNCH_BLOCKING", "TASK_QUEUE_ENABLE", "CUDA_VISIBLE_DEVICES")}
        planned = config.get("planned_inventory", {}).get("lm_trainable")
        if planned is not None and planned*4 > args.max_parameter_bytes:
            raise ValueError("planned parameter bytes exceed diagnostic bound")
        model = runtime.make_model(copy.deepcopy(config), device)
        if hasattr(model, "set_progress"):
            model.set_progress(args.progress_tokens)
        record["resolved_region_controls"] = [dict(region=index, **{
            key: selector.s.get(key) for key in ("selection_mode", "k", "allocation_temperature")})
            for index, selector in enumerate(getattr(model, "selectors", ())) if hasattr(selector, "s")]
        parameters = list(model.parameters())
        record["parameters"] = dict(unique_tensors=len(parameters),
            trainable=sum(p.numel() for p in parameters if p.requires_grad),
            bytes=sum(p.numel()*p.element_size() for p in parameters))
        if record["parameters"]["bytes"] > args.max_parameter_bytes:
            raise ValueError("actual parameter bytes exceed diagnostic bound")
        optimizer = runtime.optimizer_for(model, config)
        material = synthetic_inputs(config["training"].get("global_batch", config["training"]["batch"]),
                                    config["training"]["length"], args.input_seed)
        for _ in range(args.prime_updates):
            run_once(runtime, model, optimizer, material, config, device,
                     mode="training", progress=args.progress_tokens)
        optimizer.zero_grad(set_to_none=True)
        initial = capture(model, optimizer, device)
        record["material"] = dict(input_sha256=tensor_digest(material), initial_state_sha256=tensor_digest(initial),
            weights_sha256=tensor_digest(initial["model"]), input_shape=list(material["input_ids"].shape),
            effective_targets=int((material["labels"] != -100).sum()), input_seed=args.input_seed,
            runtime_rng_seed=config["seed"], progress_tokens=args.progress_tokens,
            private_state="independent document blocks; no cross-update private state",
            weights="frozen constructor/role-keyed seed plus explicit synthetic prime updates; no historical checkpoint",
            optimizer_initialization=("fresh empty slots; first-update state creation is timed" if not args.prime_updates
                                      else "slots and weights from real synthetic updates at the same fixed progress"),
            prime_updates=args.prime_updates, inputs="synthetic next-token IDs [4,4096)")
        record["comparison_scope"] = ("Experiment-specific core-5 direct formulas; dense physical Full then mask; "
            "static aligned regions only, no general foundation roots/continuation" if config["kind"] == "direct_semantic"
            else "Different-function frozen SmallTransformer: explicit QK^T, softmax and AV; "
                 "not the foundation FC comparison's fused-SDPA Transformer")
        write_json(output/"result.json", record)
        def step():
            return run_once(runtime, model, optimizer, material, config, device,
                            mode=args.mode, progress=args.progress_tokens)
        for _ in range(args.warmup):
            restore(initial, model, optimizer, device)
            step()
        restore(initial, model, optimizer, device)
        runtime.synchronize(device)
        if tensor_digest(capture(model, optimizer, device)) != record["material"]["initial_state_sha256"]:
            raise AssertionError("warmup changed the restored initial model/optimizer/RNG state")
        record["warmup_restored"] = True
        # CANN exporters may write scratch files in cwd; isolate them here.
        os.chdir(output)
        for index in range(args.iterations):
            if index:
                restore(initial, model, optimizer, device)
            reset_memory_peak(device)
            if args.action == "measure":
                result = step()
            else:
                from req006_profile import scopes, profile_call, summarize_trace
                from tidegraph.autograd_census import Census
                census_context = Census() if args.action == "census" else nullcontext(None)
                with census_context as census, scopes(runtime, model, optimizer, census=census):
                    if args.action == "profile":
                        if shutil.disk_usage(output).free < args.max_trace_bytes:
                            raise OSError("insufficient free filesystem space for the declared trace budget")
                        trace = output/"trace.json"
                        result = profile_call(step, device, trace)
                        if trace.stat().st_size > args.max_trace_bytes:
                            raise RuntimeError("exported trace exceeds byte budget; retained for diagnosis, no success claim")
                        record["trace"] = summarize_trace(trace)
                        record["trace"]["sha256"] = sha256(trace.read_bytes())
                        if device.type != "cpu" and not record["trace"]["kernel_events"]:
                            raise RuntimeError("requested device profile has no recorded accelerator kernels")
                    else:
                        result = step()
                        record["autograd"] = census.snapshot()
                        record["autograd_scope"] = "Returned logits and loss before each backward; exposed nodes and saved tensors. " \
                            "Custom allocation forward has no internal autograd graph; no opaque native internals are inferred."
            result["memory"] = memory(device)
            result["instrumented"] = args.action != "measure"
            if result["instrumented"]:
                for key in tuple(result):
                    if key.endswith("_per_second"):
                        del result[key]
            record["iterations"].append(result)
        record["final_model_sha256"] = tensor_digest(model.state_dict())
        record["final_optimizer_sha256"] = tensor_digest(optimizer.state_dict())
        record["loaded_downstream_files"] = {name: str(Path(module.__file__).resolve().relative_to(output/"source"))
            for name, module in sys.modules.items() if name.startswith("tide_lab") and getattr(module, "__file__", None)}
        for name, expected in record["downstream_source"]["python_files_sha256"].items():
            if sha256((output/"source"/name).read_bytes()) != expected:
                raise RuntimeError("frozen downstream source changed: "+name)
        record["state"] = "passed"
    except BaseException as error:
        record.update(state="failed", error=repr(error))
        raise
    finally:
        record["diagnostic_job_seconds"] = time.perf_counter()-started
        write_json(output/"result.json", record)
    print(json.dumps(dict(state=record["state"], action=args.action, result=str(output/"result.json"))))


if __name__ == "__main__":
    main()
