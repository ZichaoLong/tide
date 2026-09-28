"""Configuration-specific CPU equivalence gate, usable from an installed wheel."""
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from functools import partial
import math
import subprocess
import sys
import time
import torch
from .config import GraphConfig
from .execution_options import ExecutionOptions
from .library import GraphRuntime
from .records import Continuation
from .qualification_checks import compare_gradients, chunked, coverage, compare_finite
from .qualification_inputs import Probe
from .qualification_records import publish
from .qualification_training import trajectory


def qualify(config, *, device, output_dir, inputs=None, batch_size=2, positions=4,
            stop=None, input_seed=19, width=None, dtype=None, steps=3, optimizer="adamw",
            native_library=None, resume_timeout=1800, atol=None, rtol=None):
    """Validate one finite configuration; raise on any failed mandatory check.

    Optional width/dtype overrides preserve every node/edge/region/module.
    Generated inputs are labeled fixtures. CPU evidence is not backend parity.
    The caller controls thread pools and owns the new output directory.
    """
    if device not in {"cpu", "cpu:0"}:
        raise ValueError("configuration qualification currently requires explicit CPU; NPU/CUDA parity is separate")
    if type(steps) is not int or steps < 2 or optimizer not in {"adamw", "sgd", "momentum"}:
        raise ValueError("qualification requires >=2 optimizer steps and adamw/sgd/momentum")
    if type(resume_timeout) is not int or resume_timeout < 1:
        raise ValueError("resume_timeout must be a positive integer")
    requested = config if isinstance(config, GraphConfig) else GraphConfig.from_dict(config)
    effective = replace(requested, width=requested.width if width is None else width,
                        dtype=requested.dtype if dtype is None else dtype,
                        execution=replace(requested.execution, trace=True))
    atol = (1e-10 if effective.dtype == "float64" else 1e-6) if atol is None else atol
    rtol = (1e-8 if effective.dtype == "float64" else 1e-5) if rtol is None else rtol
    if any(type(t) not in (int,float) or not math.isfinite(t) or t < 0 for t in (atol,rtol)):
        raise ValueError("tolerances must be finite nonnegative numbers")
    compare = partial(compare_finite, atol=atol, rtol=rtol)
    probe = Probe(effective, inputs=inputs, batch_size=batch_size, positions=positions, stop=stop, seed=input_seed)
    oracle_options = ExecutionOptions(schedule="reference", packed=False, trace=True,
                                      mode=effective.execution.mode, zeta=effective.execution.zeta)
    if effective.execution.schedule == "reference":
        oracle_options = replace(oracle_options, schedule="streaming")
    candidate = GraphRuntime(effective, device=device, native_library=native_library)
    reference = GraphRuntime(effective, device=device, options=oracle_options)
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(schema="tide-qualification-v1", state="running", started=datetime.now(timezone.utc).isoformat(),
                  requested_config=requested.to_dict(), requested_sha256=requested.identity,
                  effective_config=effective.to_dict(), effective_sha256=effective.identity,
                  topology_unchanged=requested.graph.identity == effective.graph.identity,
                  inputs=probe.manifest(), candidate=candidate.manifest(), oracle=reference.manifest(),
                  steps=steps, optimizer=optimizer, checks=[], scope="finite CPU configuration; no general scale/backend claim",
                  probe_loss="weighted squared observables divided by participating scalar count; isolated mean-square roots",
                  tolerances={"atol":atol, "rtol":rtol})
    def checked(name, **details):
        report["checks"].append(dict(name=name, state="passed", **details))
        publish(out / "report.json", report)
    publish(out / "report.json", report)
    try:
        torch.save(dict(config=effective.to_dict(), probe=probe.payload()), out / "input.pt")
        a, b = probe.clone(), probe.clone()
        expected = a.advance(reference.session(batch_size))
        actual = b.advance(candidate.session(batch_size))
        compare(expected, actual)
        report["coverage"] = coverage(actual, candidate.graph)
        report["execution_stats"] = actual.stats
        if not actual.trace:
            raise ValueError("probe observed no graph events")
        checked("complete-observables")
        details = compare_gradients(expected, reference, a, actual, candidate, b, compare=compare)
        checked("independent-vjps", **details)
        c = probe.clone()
        parts = chunked(candidate, c)
        compare(actual, parts)
        compare_gradients(actual, candidate, b, parts, candidate, c, compare=compare)
        checked("chunk-observables-and-vjps")
        if candidate.spec:
            from .settle import run
            direct_probe = probe.clone()
            direct_runtime = GraphRuntime(effective, device="cpu", options=oracle_options)
            direct = run(direct_runtime.spec, direct_runtime.model, Continuation(candidate.graph.identity, batch_size),
                         direct_probe.values, packed=False, prefill=False, mode=oracle_options.mode, zeta=oracle_options.zeta)
            compare(expected, direct)
            compare_gradients(expected, reference, a, direct, direct_runtime, direct_probe, compare=compare)
            checked("independent-direct-settle")
        if not requested.execution.trace:
            without_trace = GraphRuntime(replace(effective, execution=replace(effective.execution, trace=False)),
                                         device=device, native_library=native_library)
            quiet = probe.clone().advance(without_trace.session(batch_size))
            compare(replace(actual, trace=[], messages=[]), quiet)
            checked("requested-trace-disabled")
        # Fresh runtimes avoid carrying any gradients or parameter updates across gates.
        ref_train = GraphRuntime(effective, device=device, options=oracle_options)
        candidate_train = GraphRuntime(effective, device=device, native_library=native_library)
        expected_steps = trajectory(ref_train, probe, steps, optimizer)
        checkpoint = out / "continuation.pt"
        actual_steps = trajectory(candidate_train, probe, steps, optimizer, checkpoint=checkpoint)
        compare(expected_steps, actual_steps)
        checked("optimizer-trajectory", steps=steps, optimizer=optimizer)
        resolved_native = str(Path(candidate.engine.core.__file__).resolve()) if candidate.engine else None
        torch.save(dict(config=effective.to_dict(), probe=probe.payload(), steps=steps, optimizer=optimizer,
                        checkpoint=str(checkpoint), native_library=resolved_native), out / "resume-input.pt")
        command = [sys.executable, "-m", "tidegraph.resume_check", str(out / "resume-input.pt"), str(out / "resumed.pt")]
        with (out / "resume.log").open("w") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=resume_timeout)
        if completed.returncode:
            raise RuntimeError(f"fresh-process resume exited {completed.returncode}; inspect {out / 'resume.log'}")
        resumed = torch.load(out / "resumed.pt", weights_only=True, map_location="cpu")
        compare(actual_steps[1:], resumed)
        checked("fresh-process-checkpoint", exit_code=completed.returncode)
        report["state"] = "passed"
    except BaseException as error:
        report.update(state="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report.update(finished=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-started)
        publish(out / "report.json", report)
    return report
