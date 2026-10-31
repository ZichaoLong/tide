"""Installed experiment execution, with independently reported acceptance."""
from pathlib import Path
import os
import statistics
import time
from .config import resolve
from .records import now, write_json


def create(resolved, *, checkpoint=None):
    if resolved["execution"]["effective"]["placement"] == "resident":
        from .resident import Resident
        return Resident(resolved, checkpoint=checkpoint)
    from .eager import Eager
    return Eager(resolved, checkpoint=checkpoint)


def _configure(resolved):
    import torch
    from ..runtime import resolve_device, manifest
    from .admission import precision
    e = resolved["execution"]["effective"]
    torch.set_num_threads(e["intraop_threads"])
    if torch.get_num_interop_threads() != e["interop_threads"]:
        torch.set_num_interop_threads(e["interop_threads"])
    devices = []
    for name in e["peer_devices"] or (e["device"],):
        device, reason = resolve_device(name)
        devices.append(device)
    device, reason = resolve_device(e["device"])
    policy = precision(device)
    from ..communication import configure
    configure(e.get("communication", "ordinary"))
    return dict(manifest(device, reason, getattr(torch, e["dtype"])), precision_policy=policy,
                actual_threads=dict(intraop=torch.get_num_threads(), interop=torch.get_num_interop_threads(), workers=e["workers"]))


def run(configuration, output, resume=None, validation_override=None, *, check_roots=False):
    resolved = resolve(configuration)
    measurement = resolved["measurement"]
    policy = measurement["validation"] if validation_override is None else validation_override
    if policy not in {"strict", "explore", "none"}:
        raise ValueError("measurement.validation: choose strict, explore or none")
    e = resolved["execution"]["effective"]
    observation_mode = measurement.get("observation", "none")
    if e.get("return_mode", "complete") != "complete" and policy != "none":
        raise ValueError("execution.return_mode: outputs/metrics runs require explicit validation=none; use tidegraph check for separate complete-record qualification")
    if e["client"] == "libtorch":
        from .standalone import run as standalone
        return standalone(resolved, output, resume=resume, validation_override=policy, check_roots=check_roots)
    from .identity import capture, verify
    identity = capture(resolved["site"]["paths"])
    checkpoint_path, start = None, 0
    if resume is not None:
        from .checkpoint import validate
        checkpoint_path, saved = validate(resume, resolved, identity)
        start = saved["next_window"]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/"resolved.json", resolved)
    write_json(output/"identity.json", identity)
    record = dict(schema="tide-experiment-result-v1", state="running", execution_status="not_started", started=now(),
        resolved_config=resolved, **{k: resolved[k] for k in ("model_id", "workload_id", "task_id", "execution_id", "config_id")},
        mode=measurement["mode"], client=e["client"], start_window=start, identity=identity,
        kind="benchmark" if observation_mode == "none" else observation_mode+"-observation",
        throughput_sample=observation_mode == "none", throughput_qualified=False,
        validation=dict(policy=policy, numeric="not_checked", discrete="not_checked", complete="not_checked", outcome="not_checked", strict_passed=False),
        checkpoint=dict(state="not_checked"), metrics=[], warmup=[],
        timing_contract=dict(return_mode=e.get("return_mode", "complete"),
            included="input preparation, transfer, host/device scheduling, packing, forward, finite checks, requested return and synchronization; training includes loss, VJP and optimizer",
            excluded="runtime/model construction, durable checkpoint publication, independent validation and recovery checks",
            no_cross_contract_speedup=True), exploratory_policy=policy == "explore")
    owner = None
    started = time.perf_counter()
    write_json(output/"result.json", record)
    try:
        from .admission import preflight
        record["runtime"] = _configure(resolved)
        record["admission"] = preflight(resolved, output, start_window=start)
        if record["admission"]["refused"]:
            raise ValueError("capacity preflight: "+"; ".join(record["admission"]["refused"]))
        from .observation import Memory
        import torch
        memory = Memory(torch.device(e["device"]), e["peer_devices"])
        record["memory"] = memory.capture("runtime_ready", reset=True)
        record["phase"] = "construction"
        write_json(output/"result.json", record)
        began = time.perf_counter()
        owner = create(resolved, checkpoint=checkpoint_path)
        record["construction_seconds"] = time.perf_counter()-began
        record["execution_status"] = "running"
        memory.capture("constructed", reset=True)
        record["effective"] = owner.manifest()
        actual = []
        total = measurement["warmup"]+measurement["steps"]
        from .checkpoint import save, exact_next
        for offset in range(total):
            index = start+offset
            record.update(phase="warmup" if offset < measurement["warmup"] else "measurement", window=index)
            write_json(output/"result.json", record)
            began = time.perf_counter()
            if observation_mode != "none" and offset >= measurement["warmup"]:
                from .profiling import step as observed_step
                (observation, loss), detail = observed_step(owner, index, observation_mode, output)
                record["observation"] = detail
            else:
                observation, loss = owner.step(index)
            elapsed = time.perf_counter()-began
            row = dict(window=index, seconds=elapsed, loss=loss, **owner.last_work)
            if e.get("return_mode", "complete") == "complete":
                from .activity import activity
                row["boundary_activity"] = activity(observation)
            record["warmup" if offset < measurement["warmup"] else "metrics"].append(row)
            if e.get("return_mode", "complete") != "metrics":
                from ..checkpoint_io import publish
                returned = observation.outputs if e.get("return_mode", "complete") == "complete" else observation["outputs"]
                filename = f"outputs-{index:06d}.pt"
                began = time.perf_counter()
                publish(output/filename, dict(window=index, outputs=returned))
                row.update(output_artifact=filename, output_write_seconds=time.perf_counter()-began)
            if policy != "none":
                actual.append((observation, loss))
            del observation
            # Durable boundaries are outside timing and individually costed.
            began = time.perf_counter()
            checkpoint_path, checkpoint = save(owner, output, resolved, identity, index+1)
            row["checkpoint_seconds"] = time.perf_counter()-began
            record["checkpoint"] = dict(state="saved", next_window=index+1, bytes=checkpoint["bytes"],
                                        path=str(checkpoint_path), exact_resume="not_checked")
            memory.capture(record["phase"], reset=offset == measurement["warmup"]-1)
            write_json(output/"result.json", record)
        record["execution_status"] = "completed"
        record["effective"] = owner.manifest()
        parameters = owner.parameters() if measurement["mode"] == "training" and policy != "none" else None
        training = owner.training_state() if measurement["mode"] == "training" and policy != "none" else None
        task_state = owner.task_state() if policy != "none" and hasattr(owner, "task_state") else None
        if measurement["checkpoint_check"]:
            record["phase"] = "exact-resume"
            write_json(output/"result.json", record)
            began = time.perf_counter()
            record["checkpoint"].update(state="passed", exact_resume=True,
                resume_check=exact_next(owner, lambda **kw: create(resolved, **kw), checkpoint_path, start+total))
            record["checkpoint"]["check_seconds"] = time.perf_counter()-began
            memory.capture("exact-resume")
        owner.close()
        owner = None
        if policy != "none":
            record["phase"] = "independent-check"
            write_json(output/"result.json", record)
            from .validation import independent, outcome
            began = time.perf_counter()
            validation = independent(resolved, actual, parameters, training, start, task_state=task_state)
            record["validation_seconds"] = time.perf_counter()-began
            record["validation"] = dict(validation, policy=policy, outcome=outcome(validation, policy))
        state = record["validation"]["outcome"]
        record["state"] = "failed" if state == "failed" else "passed" if state == "passed" else "completed"
        record["strict_passed"] = state == "passed"
        record["throughput_qualified"] = state == "passed" and observation_mode == "none" and record["checkpoint"].get("exact_resume") is True
        times = [row["seconds"] for row in record["metrics"]]
        record["timing_summary"] = dict(median_seconds=statistics.median(times), total_seconds=sum(times),
            min_seconds=min(times), max_seconds=max(times), measured_windows=len(times),
            interpretation="strict qualified complete-return" if state == "passed" else
                "exploratory finite cost; original strict comparison failed" if state == "exploratory_accepted" else "unchecked or failed cost")
        verify(identity, resolved["site"]["paths"])
        record["phase"] = "complete"
    except BaseException as error:
        record.update(state="failed", error=dict(type=type(error).__name__, message=str(error)))
        if record.get("phase") == "exact-resume":
            record["checkpoint"].update(state="failed", exact_resume=False)
        if callable(getattr(error, "record", None)):
            record["error"]["details"] = error.record()
        if record["execution_status"] != "completed":
            record["execution_status"] = "failed"
        raise
    finally:
        if owner is not None:
            owner.close()
        record.update(finished=now(), total_wall_seconds=time.perf_counter()-started, load_average_after=list(os.getloadavg()))
        write_json(output/"result.json", record)
    return record


def check(configuration, output, scope="smoke"):
    from .checks import check as verify_configuration
    return verify_configuration(configuration, output, scope)
