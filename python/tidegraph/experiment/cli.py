"""Install → inspect → plan → independently check → run/resume → compare."""
import argparse
import json
import os
from pathlib import Path
import sys
from .config import resolve


def _emit(value):
    print(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def _runtime_environment(resolved, argv):
    from .site import launch_environment
    effective = launch_environment(resolved["site"], resolved["execution"])
    if any(os.environ.get(key) != value for key, value in effective.items()):
        # The dynamic loader reads LD_LIBRARY_PATH at process startup. Merely
        # mutating os.environ after launch does not select a different SDK.
        environment = dict(os.environ)
        environment.update(effective)
        os.execve(sys.executable, [sys.executable, "-m", "tidegraph", *argv], environment)


def _common(parser, *, output=False):
    parser.add_argument("config", help="experiment JSON (public model/recipe, task, execution, measurement)")
    parser.add_argument("--site", help="tide-site-v1 JSON, relative to the current directory; overrides config.site")
    if output:
        parser.add_argument("--output-dir", required=True, help="new directory; existing results are never overwritten")


def parser():
    root = argparse.ArgumentParser(prog="tidegraph", description=__doc__,
        epilog="Planning/help/results do not import Torch. See docs/human-workflow.md for complete JSON examples.")
    from ..version import __version__
    root.add_argument("--version", action="version", version="tidegraph "+__version__)
    commands = root.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="inspect the installation; optionally run one explicit device operator")
    doctor.add_argument("--site")
    doctor.add_argument("--device", default="cpu", help="explicit cpu, cuda:N or npu:N (default cpu)")
    doctor.add_argument("--probe", choices=("metadata", "operator"), default="metadata")
    doctor.add_argument("--dtype", choices=("float16", "float32", "float64"), default="float32")
    doctor.add_argument("--disk-path", default=".")
    lock = commands.add_parser("lock", help="create/verify installed package, build and dependency identities")
    lock_actions = lock.add_subparsers(dest="action", required=True)
    create_lock = lock_actions.add_parser("create", help="record the exact current installation; optionally match a wheel")
    create_lock.add_argument("--site")
    create_lock.add_argument("--wheel")
    create_lock.add_argument("--output", required=True)
    verify_lock = lock_actions.add_parser("verify", help="rehash the installed package and selected builds")
    verify_lock.add_argument("lock_file")
    verify_lock.add_argument("--site", help="explicit relocated build paths; bytes must still match")
    plan = commands.add_parser("plan", help="allocation-free public model validation, identity and per-device budget")
    _common(plan)
    plan.add_argument("--format", choices=("json", "table"), default="table")
    plan.add_argument("--output", help="save the full resolved JSON to a new file")
    plan.add_argument("--matrix", action="store_true", help="preview every explicitly listed finite matrix case")
    check = commands.add_parser("check", help="independent base checks, separate from ordinary task execution")
    _common(check, output=True)
    check.add_argument("--scope", required=True, choices=("smoke", "related", "full"),
        help="smoke=small selected-path check; related=selected contract checks; full=declared qualification scope")
    run = commands.add_parser("run", help="one foreground inference/training run with a recorded validation policy")
    _common(run, output=True)
    run.add_argument("--resume", help="restore a complete checkpoint into this new run")
    run.add_argument("--validation", choices=("strict", "explore", "none"),
        help="explicit override; explore accepts only typed numerical mismatch, none records unchecked")
    matrix = commands.add_parser("matrix", help="run an explicit finite list serially, one bounded process per case")
    _common(matrix, output=True)
    matrix.add_argument("--detach", action="store_true", help="use the site's durable launcher for the whole matrix")
    jobs = commands.add_parser("job", help="persistent jobs: start, inspect, logs, stop, resume")
    actions = jobs.add_subparsers(dest="action", required=True)
    start = actions.add_parser("start", help="freeze configuration and installed/build identities before detaching")
    _common(start, output=True)
    start.add_argument("--resume", help="complete checkpoint to restore")
    start.add_argument("--matrix", action="store_true")
    for name, help_text in (("status", "observe real supervisor/unit state"),
                            ("logs", "read the durable task log"), ("stop", "stop all owned descendants and verify")):
        action = actions.add_parser(name, help=help_text)
        action.add_argument("run_dir")
        if name == "logs":
            action.add_argument("--lines", type=int, default=80)
    resume = actions.add_parser("resume", help="start a new persistent run from a previous job's complete checkpoint")
    resume.add_argument("run_dir")
    resume.add_argument("--output-dir", required=True)
    resume.add_argument("--checkpoint")
    results = commands.add_parser("results", help="filter result records; render terminal, CSV or standalone HTML")
    results.add_argument("paths", nargs="+")
    results.add_argument("--format", choices=("table", "json", "csv", "html", "archive"), default="table",
                         help="archive is a self-contained, queryable result projection for reviewed evidence")
    results.add_argument("--output")
    for key in ("platform", "graph", "topology", "dtype", "scoring-dtype", "client", "placement", "status",
                "qualification", "mode", "communication", "return-mode"):
        results.add_argument("--"+key)
    for key in ("batch", "params-min", "params-max"):
        results.add_argument("--"+key, type=int)
    results.add_argument("--exploratory", choices=("true", "false"))
    # Forward unchanged legacy behavior; descriptions keep it discoverable.
    commands.add_parser("inspect", help="legacy GraphConfig normalization (inspect --help)")
    commands.add_parser("qualify", help="legacy full-observable graph qualification (qualify --help)")
    return root


def _plan_table(config):
    plan, budget, mapping = config["plan"], config["capacity_budget"], config["payload_placement"]
    lines = [f"{config['name']}: model {config['model_id']}",
             f"graph={config['graph_view']['family']} nodes={plan['graph']['nodes']} "
             f"parameters={config['inventory']['totals']['trainable']} batch={plan['samples']} "
             f"positions={plan['positions']} span={plan['span']}",
             f"budget={budget['status']} automatic_admission=false; runtime peaks and hardware checks remain required"]
    for device, row in budget["per_device"].items():
        owner = mapping["per_device"].get(device, {})
        lines.append(f"  {device}: nodes={owner.get('nodes', [])} regions={owner.get('regions', [])} "
                     f"estimated_peak_bytes={row.get('estimated_peak_bytes')} usable_bytes={row.get('usable_bytes')} "
                     f"unknown={row.get('unknown_components')}")
    for refusal in budget.get("static_rejections", []):
        lines.append("  refusal: "+str(refusal))
    return "\n".join(lines)


def _terminal(record, *, cancelled=False):
    if "resolved_config" in record:
        summary = {key: record.get(key) for key in (
            "state", "execution_status", "config_id", "validation", "checkpoint", "timing_contract")}
        summary["measured_windows"] = len(record.get("metrics", []))
    elif "supervisor_pid" in record or record.get("schema") == "tide-durable-job-v1":
        summary = {key: record.get(key) for key in (
            "state", "observed_state", "started", "finished", "exit_code", "observed_supervisor_alive",
            "observed_live_workload_pids", "remaining_live_pids", "systemd", "experiment") if key in record}
        summary["task"] = {key: record.get("metadata", {}).get(key) for key in (
            "launcher", "unit", "task_kind", "config_id", "result_dir")}
    else:
        summary = record
    _emit(summary)
    state = record.get("state", "unchecked")
    return 0 if state in {"passed", "completed", "running", "starting", "metadata-only"} or (cancelled and state == "cancelled") else 1


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in {"inspect", "qualify"}:
        from ..__main__ import legacy_main
        return legacy_main(argv)
    root = parser()
    args = root.parse_args(argv)
    if getattr(args, "site", None) is not None:
        args.site = str(Path(args.site).resolve())
    try:
        if args.command == "doctor":
            if args.probe == "operator":
                from .site import load
                site = load(args.site)
                _runtime_environment(dict(site=site, execution=dict(effective=dict(
                    intraop_threads=site["threads"].get("intraop_threads", 1)))), argv)
            from .doctor import inspect
            return _terminal(inspect(site=args.site, device=args.device, probe=args.probe,
                                     dtype=args.dtype, disk_path=args.disk_path))
        if args.command == "lock":
            from .identity import create_lock, verify_lock
            if args.action == "verify":
                return _terminal(verify_lock(args.lock_file, site=args.site))
            from .records import write_json
            record = create_lock(site=args.site, wheel=args.wheel)
            write_json(args.output, record, exclusive=True)
            _emit(dict(state="recorded", lock_id=record["lock_id"], path=str(Path(args.output).resolve())))
            return 0
        if args.command == "plan":
            from .matrix import resolve_matrix
            config = resolve_matrix(args.config, site=args.site) if args.matrix else resolve(args.config, site=args.site)
            if args.output:
                from .records import write_json
                write_json(args.output, config, exclusive=True)
            if args.format == "json":
                _emit(config)
            else:
                cases = [row["config"] for row in config["cases"]] if args.matrix else [config]
                print("\n\n".join(_plan_table(row) for row in cases))
            return 0
        if args.command in {"run", "check"}:
            config = resolve(args.config, site=args.site)
            _runtime_environment(config, argv)
            from . import engine
            record = (engine.run(config, args.output_dir, resume=args.resume, validation_override=args.validation)
                      if args.command == "run" else engine.check(config, args.output_dir, scope=args.scope))
            return _terminal(record)
        if args.command == "matrix":
            from .matrix import resolve_matrix, run_matrix
            config = resolve_matrix(args.config, site=args.site)
            if args.detach:
                from .jobs import start
                return _terminal(start(config, args.output_dir, matrix=True))
            return _terminal(run_matrix(config, args.output_dir))
        if args.command == "job":
            from . import jobs
            if args.action == "start":
                return _terminal(jobs.start(args.config, args.output_dir, site=args.site, resume=args.resume, matrix=args.matrix))
            if args.action == "logs":
                print(jobs.logs(args.run_dir, lines=args.lines), end="")
                return 0
            if args.action == "resume":
                return _terminal(jobs.resume(args.run_dir, args.output_dir, checkpoint=args.checkpoint))
            return _terminal((jobs.stop if args.action == "stop" else jobs.status)(args.run_dir), cancelled=args.action == "stop")
        if args.command == "results":
            from .report import load, select, render, save
            filters = {key: value for key, value in vars(args).items()
                       if key not in {"command", "paths", "format", "output"} and value is not None}
            if "exploratory" in filters:
                filters["exploratory"] = filters["exploratory"] == "true"
            content = render(select(load(args.paths), **filters), format=args.format)
            if args.output:
                save(args.output, content)
            else:
                print(content, end="")
            return 0
        raise ValueError("unknown command")
    except (ValueError, RuntimeError, AssertionError, OSError, ImportError) as error:
        root.exit(1, f"{type(error).__name__}: {error}\n")
