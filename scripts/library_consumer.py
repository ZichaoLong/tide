#!/usr/bin/env python3
"""Build/install a wheel and consume only installed Python/native/C++ surfaces."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import venv
from durable_records import write_json
from source_identity import source_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--build-dir", default="build")
    parser.add_argument("--python-only", action="store_true", help="development scope; full consumption is the default")
    parser.add_argument("--device", default="cpu", help="explicit accelerator for target-machine consumption")
    parser.add_argument("--skip-cpp", action="store_true", help="Python-owned NPU adapters need a separate standalone SDK gate")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out, build = Path(args.output_dir).resolve(), Path(args.build_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    source, dirty = source_state(root)
    record = dict(schema="tide-consumer-v1", state="running", source=source, dirty=dirty,
                  scope="python-only" if args.python_only else "python-native" if args.skip_cpp else "python-native-cpp",
                  device=args.device, checks=[])
    write_json(out / "result.json", record)
    env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
               TORCH_DEVICE_BACKEND_AUTOLOAD="0", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env.pop("PYTHONPATH", None)
    def run(command, cwd=out):
        with (out / "commands.log").open("a") as log:
            log.write(json.dumps([str(x) for x in command]) + "\n"); log.flush()
            subprocess.run([str(x) for x in command], cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    try:
        # Setuptools may write build/egg-info: give it a staging copy, not frozen source.
        staging = out / "package"
        staging.mkdir()
        shutil.copy2(root / "pyproject.toml", staging)
        shutil.copytree(root / "python", staging / "python", ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
        run([sys.executable, "-m", "pip", "wheel", "--no-index", "--no-deps", "--no-build-isolation", staging, "-w", out / "wheels"])
        wheel, = (out / "wheels").glob("*.whl")
        record["wheel_sha256"] = hashlib.sha256(wheel.read_bytes()).hexdigest()
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(out / "env")
        python = out / "env/bin/python"
        run([python, "-m", "pip", "install", "--no-index", "--no-deps", wheel])
        consumer = out / "application"
        shutil.copytree(root / "examples/consumer", consumer)
        native = out / "native"
        if not args.python_only:
            native.mkdir()
            shutil.copy2(build / "_tide_native.so", native)
            shutil.copy2(build / "build-manifest.json", native)
        for family in ("pdg", "timed-dag", "settle"):
            config = json.loads((consumer / "graph.json").read_text())
            config["family"] = family
            config["execution"]["schedule"] = "auto"
            file = consumer / f"{family}.json"
            file.write_text(json.dumps(config, indent=2) + "\n")
            for implementation in (("python",) if args.python_only else ("python", "native")):
                result = out / f"{family}-{implementation}"
                command = [python, "train.py", "--config", file, "--device", args.device, "--implementation", implementation, "--output-dir", result]
                if implementation == "native":
                    command += ["--native-library", native]
                run(command, consumer)
                report = json.loads((result / "result.json").read_text())
                if not Path(report["library_path"]).is_relative_to(out / "env"):
                    raise RuntimeError("consumer imported outside its installed environment")
                record["checks"].append(dict(family=family, implementation=implementation, state=report["state"]))
            # Run the installed CLI + its fresh-process resume, using explicit relocated native.
            if not args.python_only:
                config["execution"]["implementation"] = "native"
                file.write_text(json.dumps(config, indent=2) + "\n")
            command = [python, "-m", "tidegraph", "qualify", file, "--device", args.device, "--output-dir", out / f"gate-{family}",
                       "--steps", "2", "--positions", "2"]
            if not args.python_only:
                command += ["--native-library", native]
            run(command, consumer)
            gate = json.loads((out / f"gate-{family}/report.json").read_text())
            if gate["state"] != "passed":
                raise RuntimeError("installed qualification failed")
            record["checks"].append(dict(implementation="installed-qualifier", family=family,
                                         state=gate["state"], checks=len(gate["checks"])))
        if not args.python_only and not args.skip_cpp:
            run(["cmake", "--install", build, "--prefix", out / "prefix"])
            shutil.copytree(root / "examples/consumer_cpp", out / "cpp-application")
            import torch
            run(["cmake", "-S", out / "cpp-application", "-B", out / "cpp-build", "-G", "Ninja",
                 "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_PREFIX_PATH={out / 'prefix'};{torch.utils.cmake_prefix_path}"])
            run(["cmake", "--build", out / "cpp-build", "--parallel", "2"])
            dtypes = ["float32", "float64"] if args.device == "cpu" else ["float32"]
            for dtype in dtypes:
                run([out / "cpp-build/consumer", "--device", args.device, "--dtype", dtype])
            record["checks"].append(dict(implementation="installed-cpp", state="passed", dtypes=dtypes))
        if source_state(root) != (source, dirty):
            raise RuntimeError("source changed during consumer verification")
        record["state"] = "passed"
    except BaseException as error:
        record.update(state="failed", error=repr(error))
        raise
    finally:
        write_json(out / "result.json", record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
