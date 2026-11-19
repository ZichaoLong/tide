"""Freeze an exact commit and submit a bounded background.slice experiment.

Example: python scripts/launch.py --name cpu-main --threads 8 --affinity 0-7 \
  -- python -m benchmarks.run --config configs/fc-k.json --output RESULT.json
Relative output arguments are relative to the immutable source; prefer an
absolute output under the new run directory, or the literal {run} placeholder.
"""
import argparse
import hashlib
import io
import os
from pathlib import Path
import shlex
import subprocess
import tarfile

from job import atomic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--affinity", required=True)
    parser.add_argument("--npu-count", type=int, default=0)
    parser.add_argument("--candidate-ids")
    parser.add_argument("--seconds", type=int, default=3600)
    parser.add_argument("--rss-gib", type=float, default=64)
    parser.add_argument("--queue-seconds", type=int, default=600)
    parser.add_argument("--after-unit", action="append", default=[],
                        help="wait for this user unit to become inactive before CPU work/device admission")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in args.name):
        parser.error("name must contain only lower-case ASCII letters, digits and hyphens")
    if min(args.threads, args.seconds, args.queue_seconds, args.rss_gib) <= 0 or args.npu_count < 0:
        parser.error("resource and time bounds must be positive")
    root = Path(__file__).resolve().parents[1]
    source = subprocess.check_output(["git", "rev-parse", args.revision], cwd=root, text=True).strip()
    run = root / "artifacts" / args.name
    run.mkdir(parents=True, exist_ok=False)
    frozen = run / "source"
    frozen.mkdir()
    archive = subprocess.check_output(["git", "archive", source], cwd=root)
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        stream.extractall(frozen, filter="data")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("provide a workload after --")
    python = str(root / ".venv-instances/bin/python")
    command = [item.replace("{run}", str(run)).replace("{source}", str(frozen)) for item in command]
    if command[0] == "python":
        command[0] = python
    # The site helper is external read-only infrastructure; record its path.
    queue = Path("/mi/data2T/zlong/home/.agents/skills/use-local-ascend/scripts/run_on_free_npu.py")
    if args.npu_count:
        admission = [python, str(queue), "--job-id", args.name, "--project", "tide-instances",
                     "--priority", "0", "--npu-count", str(args.npu_count), "--max-wait-seconds",
                     str(args.queue_seconds), "--state-file", str(run / "queue.json")]
        if args.candidate_ids:
            admission += ["--candidate-ids", args.candidate_ids]
        command = admission + ["--"] + command
    if args.after_unit:
        command = [python, str(frozen / "scripts/wait_units.py"),
                   "--status", str(run / "dependency.json"), "--seconds", str(args.seconds),
                   *[item for unit in args.after_unit for item in ("--unit", unit)], "--", *command]
    supervised = [python, str(frozen / "scripts/job.py"), "--status", str(run / "status.json"),
                  "--source", source, "--max-rss-gib", str(args.rss_gib), "--"] + command
    launcher = run / "run.sh"
    lines = ["#!/bin/bash", "set -eo pipefail", "source /usr/share/Modules/init/bash",
             "module purge", "unset PYTHONPATH", "module load ascend/dev-workspace-9.2.0",
             "source " + shlex.quote(str(root / ".venv-instances/bin/activate"))]
    environment = {"TIDE_SOURCE_COMMIT": source,
                   "TIDE_HARNESS_COMMIT": source, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0" if not args.npu_count else "1",
                   "TIDE_RESOURCE_IDENTITY": "exclusive_cooperative_queue; external jobs not controlled" if args.npu_count else "pinned_cpu; shared host",
                   "OMP_NUM_THREADS": str(args.threads), "MKL_NUM_THREADS": str(args.threads),
                   "OPENBLAS_NUM_THREADS": str(args.threads)}
    lines += ["export " + key + "=" + shlex.quote(value) for key, value in environment.items()]
    # The selected module supplies CANN's TBE Python packages. Replacing its
    # search path breaks lazy operator compilation even when eager ops work.
    # Prepend our frozen package without inheriting a different caller stack.
    lines += ["export PYTHONPATH=" + shlex.quote(str(frozen / "src")) + '"${PYTHONPATH:+:$PYTHONPATH}"']
    environment_record = "\n".join([
        "import importlib.util, os, sys",
        "from scripts.job import atomic",
        "names = ('PYTHONPATH', 'PATH', 'LD_LIBRARY_PATH', 'ASCEND_HOME_PATH', 'ASCEND_OPP_PATH', 'LOADEDMODULES', 'VIRTUAL_ENV', 'TORCH_DEVICE_BACKEND_AUTOLOAD')",
        "modules = {name: importlib.util.find_spec(name) for name in ('tide', 'tbe')}",
        "atomic(sys.argv[1], dict(executable=sys.executable, sys_path=sys.path, environment={name: os.environ.get(name) for name in names}, module_origins={name: None if spec is None else spec.origin for name, spec in modules.items()}))",
    ])
    lines += [shlex.join([python, "-c", environment_record, str(run / "environment.json")])]
    lines += ["exec " + shlex.join(supervised), ""]
    launcher.write_text("\n".join(lines))
    unit = "tide-" + args.name
    launch = ["systemd-run", "--user", "--unit=" + unit, "--service-type=exec", "--slice=background.slice",
              "--working-directory=" + str(frozen), "--property=KillMode=control-group",
              "--property=RuntimeMaxSec=" + str(args.seconds), "--property=TimeoutStopSec=30",
              "--property=CPUAffinity=" + args.affinity,
              "--property=StandardOutput=append:" + str(run / "stdout.log"),
              "--property=StandardError=append:" + str(run / "stderr.log"), "/bin/bash", str(launcher)]
    record = dict(source=source, source_path=str(frozen), run=str(run), unit=unit, arguments=vars(args),
                  launch=launch, command=command, environment=environment, state="prepared",
                  pythonpath_policy=dict(prepend=str(frozen / "src"), inherit="selected baseline module only; caller PYTHONPATH cleared before module load"),
                  actual_environment_path=str(run / "environment.json"),
                  launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  resource_enforcement="process affinity; sampled process-tree RSS, not cgroup hard memory",
                  device_policy="exclusive cooperative whole-group queue" if args.npu_count else "cpu")
    atomic(run / "launch.json", record)
    subprocess.run(launch, check=True)
    observation = subprocess.check_output(["systemctl", "--user", "show", unit, "-p", "Id", "-p", "MainPID",
        "-p", "ControlGroup", "-p", "Slice", "-p", "ActiveState", "-p", "SubState"], text=True)
    record.update(state="submitted", unit_observation=observation)
    atomic(run / "launch.json", record)
    print(observation, end="")
    print(run)


if __name__ == "__main__":
    main()
