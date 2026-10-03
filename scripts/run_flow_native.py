"""Language-neutral launcher for the independently linked LibTorch consumer."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
from durable_records import replace_text
from flow_protocol import native_text
from flow_resident_options import native_arguments
from flow_failure import RecordedFailure


def run(packet, args):
    binary = args.native_binary.resolve()
    if not binary.is_file():
        raise ValueError("explicit standalone binary is unavailable")
    if args.native_library is not None:
        raise ValueError("standalone LibTorch does not load a Python native_library")
    # This text is derived afresh from the hash-validated JSON. No numerical
    # reference, parameters, outputs, events or gradients are prepared here.
    path = args.output_dir / "topology.txt"
    text = native_text(packet)
    replace_text(path, text)
    command = [str(binary), "--packet="+str(path.resolve()),
               "--output-dir="+str((args.output_dir/"consumer").resolve()),
               "--device="+args.device, "--dtype="+args.dtype, "--family="+args.family,
               "--preset="+args.preset, "--schedule="+args.schedule, "--optimizer="+args.optimizer,
               "--steps="+str(args.steps), "--warmup="+str(args.warmup),
               "--windows-per-step="+str(args.windows_per_step), "--threads="+str(args.threads),
               "--workers="+str(args.workers),
               "--sample-chunk-rows="+str(args.sample_chunk_rows),
               "--parameter-budget="+str(args.parameter_budget)]
    if getattr(args,"loss_scale",1.) != 1:
        command.append("--loss-scale="+str(args.loss_scale))
    if args.packed_sources: command.append("--packed-sources")
    if args.batch_next: command.append("--batch-next")
    for name in ("read", "control", "selection", "events", "scoring_dtype"):
        command.append("--"+name.replace("_", "-")+"="+getattr(args,name))
    if args.training: command.append("--training")
    if args.diagnostics: command.append("--diagnostics")
    if getattr(args,"phase_timing",False): command.append("--phase-timing")
    command.extend(native_arguments(args))
    start = time.perf_counter()
    with (args.output_dir/"consumer.log").open("w") as stream:
        process = subprocess.run(command, cwd=args.output_dir, stdout=stream, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter()-start
    identity = dict(packet_identity="hash-validated JSON; exact derived v2 text",
                    binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                    native_input_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    command=command, process_wall_seconds=elapsed)
    if path.read_text() != text:
        raise ValueError("native input packet changed during execution")
    if process.returncode:
        message = f"standalone consumer failed ({process.returncode}); see consumer.log"
        try:
            failed = json.loads((args.output_dir/"consumer/result.json").read_text())
        except (OSError, ValueError):
            failed = {}
        if isinstance(failed,dict) and failed.get('state') == 'failed' and failed.get('workload_sha256') == packet['sha256']:
            raise RecordedFailure(failed.get('error', message),
                dict(failed, **identity, native_exit_code=process.returncode))
        raise RuntimeError(message)
    result = json.loads((args.output_dir/"consumer/result.json").read_text())
    if result.get("state") != "passed" or result.get("workload_sha256") != packet["sha256"]:
        raise ValueError("standalone result identity/state mismatch")
    result.update(identity)
    return result
