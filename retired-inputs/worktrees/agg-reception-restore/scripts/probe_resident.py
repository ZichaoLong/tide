"""Bounded fullgraph feasibility probe for actual data-dependent Tide forward.

This is a capture probe, not a resident runtime implementation. It never feeds
precomputed routing to the model or substitutes dense Full. Each isolated
attempt gets two fixed inputs and a finite wall-clock limit. TorchAir is used
on NPU when installed; otherwise backend=eager supplies frontend-only evidence.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from scripts.job import ProcessTree, atomic, enable_subreaper, stop_group


def run_bounded(command, *, stdout, stderr, seconds, supervisor_pid=None,
                terminate_grace=5, record_path=None):
    """Bound one workload, retaining ownership across orphaning and setsid.

    A dedicated runner enables subreaping before passing its own supervisor PID.
    Ordinary callers may omit that seed; they must not create instant orphans.
    """
    started = time.monotonic()
    child = tracker = None
    record = {"state": "starting", "return_code": None, "timed_out": False,
              "cleanup_complete": True, "sent_sigkill": False,
              "stdout": str(stdout), "stderr": str(stderr),
              "peak_rss_bytes": 0, "max_processes": 0,
              "descendant_supervision": "retained PID/start identities; Linux subreaper" if supervisor_pid is not None
                                        else "retained PID/start identities; no orphan adoption seed"}
    def persist():
        if record_path is not None:
            atomic(record_path, record)
    persist()
    try:
        with Path(stdout).open("x") as output, Path(stderr).open("x") as errors:
            child = subprocess.Popen(command, stdout=output, stderr=errors, start_new_session=True)
            tracker = ProcessTree(child.pid, supervisor_pid=supervisor_pid)
            record.update(state="running", pid=child.pid, cleanup_complete=False)
            persist()
            deadline = started + seconds
            while True:
                code = child.poll()
                rss, count = tracker.rss()
                record.update(peak_rss_bytes=max(record["peak_rss_bytes"], rss),
                              max_processes=max(record["max_processes"], count))
                if code is not None and count == 0:
                    record.update(state="completed" if code == 0 else "worker_failed",
                                  return_code=code, cleanup_complete=True)
                    break
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    record.update(state="timed_out", timed_out=True)
                    break
                time.sleep(min(.1, remaining))
    except BaseException as error:
        record.update(state="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                      error_type=type(error).__name__, error=repr(error))
    finally:
        if child is not None:
            try:
                if tracker is None:
                    tracker = ProcessTree(child.pid, supervisor_pid=supervisor_pid)
                if child.poll() is None or tracker.refresh():
                    escalated, gone = stop_group(child, terminate_grace, tracker)
                    record.update(sent_sigkill=escalated, cleanup_complete=gone)
                else:
                    record["cleanup_complete"] = True
                record["return_code"] = child.returncode
                record["descendant_identities"] = tracker.record()
                if supervisor_pid is not None:
                    # Reap only the identities owned by this workload. The
                    # root's Popen already owns its wait status.
                    for item in record["descendant_identities"]:
                        if item["pid"] != child.pid:
                            try:
                                os.waitpid(item["pid"], os.WNOHANG)
                            except ChildProcessError:
                                pass
            except BaseException as error:
                record.update(cleanup_complete=False, cleanup_error=repr(error))
        record["elapsed_seconds"] = time.monotonic()-started
        persist()
    return record


def host_decision_sites():
    import tide
    root = Path(tide.__file__).parent
    rows = []
    for name in ("model.py", "attention.py", "allocation.py", "layout.py", "parameters.py", "kv.py"):
        for number, line in enumerate((root/name).read_text().splitlines(), 1):
            if any(token in line for token in ("bool(", "nonzero(", "int(counts.max", ".tolist()")):
                rows.append({"file": name, "line": number, "expression": line.strip()})
    return rows


def worker(args):
    import torch
    if args.device.startswith("npu"):
        import torch_npu
        torch.npu.matmul.allow_hf32 = False
        torch.npu.conv.allow_hf32 = False
    from tide import ModelConfig, TideModel
    from tide.precision import precision_context
    from scripts.qualify_precision import comparison, cpu_tree, fields_of, fingerprint
    device = torch.device(args.device)
    if device.type == "npu":
        torch.npu.set_device(device)
    torch.set_num_threads(args.threads)
    config_path = Path(args.worker_config)
    raw = json.loads(config_path.read_text())
    raw["d"] = args.width
    cfg = ModelConfig.from_dict(raw)
    torch.manual_seed(181118)
    model = TideModel(cfg)
    x = torch.randn(args.batch, args.length, cfg.input_dim or cfg.d)
    source = os.environ.get("TIDE_SOURCE_COMMIT") or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    record = {"source": source, "config": cfg.to_dict(), "device": str(device), "precision": args.precision,
              "initial_parameters_sha256": fingerprint(model.state_dict()), "input_sha256": [fingerprint(x), fingerprint(-x)],
              "torch": str(torch.__version__), "threads": torch.get_num_threads(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
              "physical_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"), "setting": args.setting,
              "state": "running", "scope": "TideModel.forward through Result/private state; comparisons/readout outside capture",
              "resident_qualified": False, "precomputed_route_supplied": False, "dense_full_substitution": False,
              "host_decision_lexical_candidates": host_decision_sites(),
              "host_decision_scan_scope": "lexical candidates include construction-only sites; executed traceback owns actual blocker",
              "backend_invocations": 0}
    atomic(args.output, record)
    try:
        model = model.to(device).eval()
        values = [x.to(device), (-x).to(device)]
        references, audits = [], []
        for value in values:
            with torch.no_grad(), precision_context(device, args.precision):
                result = model(value, record_work=True)
            references.append(cpu_tree(fields_of(result)))
            audits.append({"work": result.work, "output_support_sha256": fingerprint(result.strengths > 0)})
        record["eager_inputs"] = audits
        record["distinct_observed_routes"] = (audits[0]["work"] != audits[1]["work"] or
                                               audits[0]["output_support_sha256"] != audits[1]["output_support_sha256"])
        spec = importlib.util.find_spec("torchair")
        record["torchair_discovery"] = None if spec is None else spec.origin
        if device.type == "npu" and spec is not None:
            import torchair
            record["backend"] = "torchair"
            record["evidence_scope"] = "actual_NPU_backend_if_frontend_reaches_it"
            factory = getattr(torchair, "get_npu_backend", None)
            if factory is None:
                raise RuntimeError("installed torchair has no get_npu_backend entry")
            implementation = factory()
        else:
            from torch._dynamo.backends.registry import lookup_backend
            implementation = lookup_backend("eager")
            record["backend"] = "eager"
            record["evidence_scope"] = "frontend_only; no resident device backend certified"
            record["backend_reason"] = "TorchAir not installed" if spec is None else "CPU probe; TorchAir requires NPU"
        def backend(graph, inputs):
            record["backend_invocations"] += 1
            record.setdefault("captured_graph_sha256", []).append(hashlib.sha256(graph.code.encode()).hexdigest())
            return implementation(graph, inputs)
        settings = {"capture_scalar_outputs": args.setting == "dynamic_outputs",
                    "capture_dynamic_output_shape_ops": args.setting == "dynamic_outputs"}
        record["dynamo_settings"] = settings
        atomic(args.output, record)
        with torch._dynamo.config.patch(**settings):
            compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=True)
            checks = []
            for value, reference in zip(values, references):
                with torch.no_grad(), precision_context(device, args.precision):
                    result = compiled(value)
                if device.type == "npu":
                    torch.npu.synchronize(device)
                elif device.type == "cuda":
                    torch.cuda.synchronize(device)
                checks.append(comparison(fields_of(result), reference, {"atol": 1e-6, "rtol": 2e-3}))
            record["comparisons"] = checks
        record["state"] = "captured" if all(not c["metadata_failures"] and not c["nonfinite"] and
                                             not c["support_differences"] and not c["threshold_failures"] for c in checks) else "captured_with_deviation"
        record["capture_limit"] = "two complete calls; host invocation remains; no autonomous pending/next-step resident scheduler is implemented"
    except Exception as error:
        record.update(state="capture_blocked", error_type=type(error).__name__, error=str(error)[:12000],
                      failure_phase="frontend_or_backend_setup" if record["backend_invocations"] == 0 else "backend_or_execution")
        traceback.print_exc()
    atomic(args.output, record)
    return 0  # A recorded capture blocker is a completed probe, never a resident pass.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--configs", nargs="+", default=["configs/fc-k.json", "configs/hb-lattice-k.json"])
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--length", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=60)
    parser.add_argument("--settings", nargs="+", choices=("strict", "dynamic_outputs"), default=["strict", "dynamic_outputs"])
    parser.add_argument("--worker-config", help=argparse.SUPPRESS)
    parser.add_argument("--setting", choices=("strict", "dynamic_outputs"), help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not math.isfinite(args.timeout_seconds) or min(args.width, args.batch, args.length, args.threads, args.timeout_seconds) <= 0 or args.width % 64:
        parser.error("new output, positive bounds, and width divisible by 64 are required")
    if args.worker_config:
        return worker(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    attempts = args.output.with_name(args.output.stem + "-attempts")
    attempts.mkdir(exist_ok=False)
    record = {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "exit_policy": "probe completion, including explicit capture blockers, is not resident qualification",
              "resident_qualified": False, "rows": []}
    # This serial driver owns every child it starts. Subreaping preserves
    # ownership if a compiler helper starts a new session and loses its parent.
    enable_subreaper()
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    stop = False
    for config in args.configs:
        for setting in args.settings:
            stem = Path(config).stem + "-" + setting
            output = attempts/(stem + ".json")
            command = [sys.executable, "-m", "scripts.probe_resident", "--device", args.device, "--worker-config", config,
                       "--setting", setting, "--precision", args.precision, "--width", str(args.width),
                       "--batch", str(args.batch), "--length", str(args.length), "--threads", str(args.threads), "--output", str(output)]
            lifecycle = run_bounded(command, stdout=attempts/(stem + ".stdout.log"), stderr=attempts/(stem + ".stderr.log"),
                                    seconds=args.timeout_seconds, supervisor_pid=os.getpid(),
                                    record_path=attempts/(stem + ".lifecycle.json"))
            result = json.loads(output.read_text()) if output.exists() else None
            row = {"config": config, "setting": setting, "command": command, "lifecycle": lifecycle, "result": result,
                   "state": "cleanup_incomplete" if not lifecycle["cleanup_complete"] else
                            "interrupted" if lifecycle["state"] == "interrupted" else
                            "timeout" if lifecycle["timed_out"] else
                            "worker_failed" if lifecycle["state"] == "failed" or lifecycle["return_code"] else
                            result["state"] if result else "missing_result"}
            record["rows"].append(row)
            atomic(args.output, record)
            print(json.dumps({"config": config, "setting": setting, "state": row["state"]}), flush=True)
            if row["state"] in {"cleanup_incomplete", "interrupted"}:
                stop = True
                break
        if stop:
            break
    return int(any(row["state"] in {"worker_failed", "missing_result", "timeout", "cleanup_incomplete", "interrupted"}
                   for row in record["rows"]))


if __name__ == "__main__":
    raise SystemExit(main())
