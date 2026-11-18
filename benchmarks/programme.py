"""Finite serial benchmark suites; run inside the project's durable launcher.

Each child owns its output and log, has a time bound and persists a command/exit
record. Device admission and process-tree RAM bounds belong to scripts/launch.py
and scripts/job.py; this driver never allocates or claims exclusive hardware.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from scripts.job import ProcessTree, atomic, enable_subreaper, stop_group


REPRESENTATIVE = ["configs/"+name+".json" for name in (
    "fc-k", "fc-s", "hb-lattice-k", "hb-exchange-x", "hb-multiscale-x-settle", "hb-multiscale-x-timed", "fc-k-149m")]
DEFAULT_SUBJECTS = ["tide", "transformer_A", "transformer_B", "transformer_C", "transformer_D"]


def _now():
    return datetime.now(timezone.utc).isoformat()


def build_plan(args, output_dir):
    configs = args.configs or (REPRESENTATIVE if args.suite == "representative" else [REPRESENTATIVE[0]])
    if args.suite == "scale" and not args.configs:
        raise ValueError("scale requires explicit --configs and resource budgeting before submission")
    cases = [(Path(path).stem, Path(path), args.batch, args.length, "natural") for path in configs]
    if args.suite in {"sparse", "combinations", "shape"}:
        if len(configs) != 1:
            raise ValueError("sparse/combinations/shape use one base config to avoid a Cartesian sweep")
        source = json.loads(Path(configs[0]).read_text())
        generated = output_dir / "configs"
        generated.mkdir(exist_ok=True)
        def config(name, **overrides):
            value = json.loads(json.dumps(source))
            topology = overrides.pop("topology", None)
            value.update(overrides)
            if topology:
                value["topology"].update(topology)
            path = generated / (name+".json")
            if path.exists():
                raise FileExistsError(path)
            atomic(path, value)
            return path
        # Each suite changes one planned dimension or a small named pairing.
        if args.suite == "sparse":
            if source["topology"]["kind"] != "fc":
                raise ValueError("sparse suite requires FC base configuration")
            path = config("fc-n32-quad", topology=dict(n=32), allocation="quad", budget=1)
            cases = [(activity, path, args.batch, args.length, activity)
                     for activity in args.activities]
        elif args.suite == "combinations":
            cases = [("base", Path(configs[0]), args.batch, args.length, "natural")]
            for name, options in (
                ("light64-quad-regional", dict(aggregation="incoming_light_64", allocation="quad", sharing="region_shared")),
                ("wide-shannon-layer", dict(aggregation="incoming_wide", allocation="shannon", sharing="layer_shared")),
                ("source-pool-independent", dict(aggregation="source_residual", allocation="pool", sharing="independent"))):
                cases.append((name, config(name, **options), args.batch, args.length, "natural"))
        else:
            cases = [("base", Path(configs[0]), args.batch, args.length, "natural"),
                     ("batch4x", Path(configs[0]), args.batch*4, args.length, "natural"),
                     ("length4x", Path(configs[0]), args.batch, args.length*4, "natural")]
            n = source["topology"].get("n", 8)
            if n < 128:
                cases.append(("nodes2x", config("nodes2x", topology=dict(n=min(128, 2*n))), args.batch, args.length, "natural"))
    plan = []
    transformer_specs = [] if args.continuous_transformers is None else json.loads(Path(args.continuous_transformers).read_text())
    for label, path, batch, length, activity in cases:
        configuration = json.loads(path.read_text())
        config_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        for precision in args.precision:
            common = ["--device", args.device, "--precision", precision, "--threads", str(args.threads),
                      "--batch", str(batch), "--warmup", str(args.warmup)]
            commands = []
            if args.suite == "continuous":
                for mode in args.modes:
                    base = ["--config", str(path.resolve()), *common, "--mode", mode,
                            "--prefill", str(length), "--prefill-chunk", str(args.prefill_chunk),
                            "--decode", str(args.decode), "--updates", str(args.steps)]
                    commands.append(("tide-"+mode, "benchmarks.continuous", [*base, "--subject", "tide"], mode == "train", None))
                    for joint in args.continuous_joint:
                        commands.append(("joint-"+joint+"-"+mode, "benchmarks.continuous",
                                         [*base, "--subject", "transformer", "--joint-match", joint],
                                         False, dict(joint_class=joint, reference="initial audit of this exact stream in child")))
                    for spec in transformer_specs:
                        flags = ["--subject", "transformer", "--transformer-width", str(spec["d"]),
                                 "--transformer-layers", str(spec["layers"]), "--ffn-width", str(spec["ffn_width"]),
                                 "--head-dim", str(spec.get("head_dim", 32))]
                        commands.append((spec["label"]+"-"+mode, "benchmarks.continuous", [*base, *flags], False, spec))
            elif args.suite == "hotspots":
                for mode in args.modes:
                    commands.append((mode, "benchmarks.hotspots", ["--config", str(path.resolve()), *common,
                                      "--mode", mode, "--length", str(length), "--steps", str(args.steps)], False, None))
            elif args.suite == "projections":
                for kind in ("head_query", "ffn"):
                    for mode in args.modes:
                        commands.append((kind+"-"+mode, "benchmarks.projections", [*common, "--mode", mode,
                                         "--kind", kind, "--d", str(configuration["d"]), "--nodes", str(configuration["topology"].get("n", 8)),
                                         "--sharing", "independent" if configuration.get("sharing", "independent") == "independent" else "shared",
                                         "--length", str(length), "--steps", str(args.steps)], False, None))
            else:
                subjects = list(args.subjects or DEFAULT_SUBJECTS)
                if configuration.get("memory") == "S" and configuration["topology"]["kind"] == "fc" and "ssd_parameters" not in subjects:
                    subjects.append("ssd_parameters")
                commands.append(("all-modes", "benchmarks.run", ["--config", str(path.resolve()), *common,
                                 "--length", str(length), "--steps", str(args.steps), "--modes", *args.modes,
                                 "--subjects", *subjects, "--activity", activity], False, None))
            for suffix, module, flags, checkpoint, reference in commands:
                identity = f"{len(plan):03d}-{label}-{precision}-{suffix}"
                result = output_dir/(identity+".json")
                command = [sys.executable, "-u", "-m", module, *flags, "--output", str(result)]
                if checkpoint and args.checkpoints:
                    command.extend(["--checkpoint", str(output_dir/(identity+".pt"))])
                plan.append(dict(identity=identity, state="planned", command=command,
                                 config=str(path.resolve()), config_sha256=config_hash,
                                 output=str(result), log=str(output_dir/(identity+".log")),
                                 timeout_seconds=args.child_timeout, continuous_transformer_reference=reference))
    return plan


def run_child(row, persist, *, supervisor_pid=None):
    row.update(state="running", started=_now())
    persist()
    start, child, tracker = time.monotonic(), None, None
    try:
        with open(row["log"], "xb", buffering=0) as log:
            child = subprocess.Popen(row["command"], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            tracker = ProcessTree(child.pid, supervisor_pid=supervisor_pid)
            row["pid"] = child.pid
            persist()
            try:
                deadline = time.monotonic()+row["timeout_seconds"]
                while True:
                    code = child.poll()
                    _, count = tracker.rss()
                    if code is not None and count == 0:
                        break
                    remaining = deadline-time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(row["command"], row["timeout_seconds"])
                    time.sleep(min(.1, remaining))
                row.update(exit_code=code, state="completed" if code == 0 else "failed")
            except subprocess.TimeoutExpired:
                escalated, gone = stop_group(child, 5, tracker)
                row.update(state="timed_out", exit_code=child.returncode, escalated=escalated, descendants_gone=gone)
        path = Path(row["output"])
        if path.exists():
            result = json.loads(path.read_text())
            row["inner_state"] = result.get("state", "no_top_level_state")
            bad = result.get("state") in {"failed", "error", "running"} or any(
                item.get("finite") is False or item.get("state") in {"failed", "error", "running"}
                for key in ("rows", "observations") for item in result.get(key, []))
            if bad and row["state"] == "completed":
                row["state"] = "inner_failed"
            row["result_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        elif row["state"] == "completed":
            row["state"] = "missing_output"
    except BaseException as error:
        if child is not None and (child.poll() is None or tracker is not None and tracker.refresh()):
            escalated, gone = stop_group(child, 5, tracker)
            row.update(escalated=escalated, descendants_gone=gone, exit_code=child.returncode)
        row.update(state="cancelled" if isinstance(error, KeyboardInterrupt) else "failed", error=repr(error))
        if isinstance(error, KeyboardInterrupt):
            raise
    finally:
        if tracker is not None:
            row["descendant_identities"] = tracker.record()
        row.update(ended=_now(), elapsed_seconds=time.monotonic()-start)
        persist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("representative", "continuous", "sparse", "combinations", "shape", "scale", "hotspots", "projections"), default="representative")
    parser.add_argument("--configs", nargs="+")
    parser.add_argument("--precision", "--precisions", nargs="+", choices=("fp32", "bf16"), default=["fp32", "bf16"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--modes", nargs="+", choices=("inference", "grad_forward", "train"), default=["inference", "grad_forward", "train"])
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--activities", nargs="+", choices=("natural", "sparse03", "sparse10", "high"), default=["natural", "sparse03", "sparse10", "high"])
    parser.add_argument("--prefill-chunk", type=int, default=64)
    parser.add_argument("--decode", type=int, default=8)
    parser.add_argument("--continuous-transformers", help="JSON list of explicit d/layers/ffn_width/head_dim/label specs, each with its reference identity")
    parser.add_argument("--continuous-joint", nargs="*", choices=("A", "B", "C", "D"), default=[],
                        help="joint matches built from each child's exact initial stream, not reused from fresh-prefill records")
    parser.add_argument("--checkpoints", action="store_true", help="save and test one restore after each continuous Tide train run")
    parser.add_argument("--child-timeout", type=float, default=1800)
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.child_timeout) or min(args.batch, args.length, args.threads, args.steps, args.prefill_chunk, args.child_timeout) <= 0 or min(args.warmup, args.decode) < 0:
        parser.error("positive dimensions, steps and child timeout; nonnegative warmup and decode required")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = output/"programme.json"
    if manifest.exists():
        parser.error("programme already exists; select a new output identity")
    try:
        plan = build_plan(args, output)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    source = os.environ.get("TIDE_SOURCE_COMMIT") or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    record = dict(schema="tide-programme-v1", state="planned", source=source,
                  arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  cwd=os.getcwd(), cpu_affinity=sorted(os.sched_getaffinity(0)),
                  physical_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                  resource_identity=os.environ.get("TIDE_RESOURCE_IDENTITY", "unspecified"),
                  scope="serial independent child processes; completion is execution status, not a numerical-equivalence pass",
                  children=plan)
    def persist():
        atomic(manifest, record)
    persist()
    if args.plan_only:
        print(json.dumps(dict(state="planned", children=len(plan), manifest=str(manifest))))
        return 0
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    record.update(state="running", started=_now())
    persist()
    try:
        enable_subreaper()
        record["descendant_supervision"] = "Linux child subreaper; PID/start tracking including nested sessions"
        for row in plan:
            run_child(row, persist, supervisor_pid=os.getpid())
            print(json.dumps({key: row[key] for key in ("identity", "state", "elapsed_seconds")}), flush=True)
            if args.stop_on_failure and row["state"] != "completed":
                break
        record["state"] = "completed" if all(row["state"] == "completed" for row in plan) else "failed"
    except KeyboardInterrupt:
        record["state"] = "cancelled"
    except Exception as error:
        record.update(state="failed", error=repr(error))
    finally:
        record["ended"] = _now()
        persist()
    return int(record["state"] != "completed")


if __name__ == "__main__":
    raise SystemExit(main())
