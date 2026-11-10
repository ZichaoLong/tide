#!/usr/bin/env python3
"""TL-REQ-004 controlled cost probe using the frozen downstream-owned task.

Read Git objects from --downstream-repo; never alter its checkout/install/runs.
Run each configuration in a fresh bounded service/process. No learning claim.
"""
import argparse
import copy
import functools
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
from durable_records import write_json

DOWNSTREAM = "0439efd2af31cf10b07ce9e6423b0195f7449968"
CONFIG = "configs/studies/language/v6/scale-fc16-kv-quad.json"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--downstream-repo", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "npu:0"), required=True)
    p.add_argument("--client", choices=("python", "native"), default="native")
    p.add_argument("--query-execution", choices=("auto", "legacy", "dense-v1", "packed-v2"), default="auto")
    p.add_argument("--prefill-schedule", choices=("auto", "greedy", "window"), default="auto")
    p.add_argument("--native-library")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--length", type=int, default=128)
    p.add_argument("--profile", choices=("none", "cpu", "npu"), default="none")
    p.add_argument("--warmup-state", choices=("reset", "continue"), default="reset",
                   help="reset restores initial weights/optimizer after warmup for controlled comparisons")
    a = p.parse_args()
    if a.client == "native" and not a.native_library:
        p.error("native requires an explicit matching --native-library build directory")
    if not (1 <= a.batch <= 8 and 1 <= a.length <= 128):
        p.error("this bounded probe requires 1<=B<=8, 1<=L<=128")
    out = a.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    source = out/"downstream-source"; source.mkdir()
    archive = subprocess.check_output(["git", "-C", str(a.downstream_repo), "archive", DOWNSTREAM, "tide_lab", CONFIG])
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(source, filter="data")
    sys.path.insert(0, str(source))
    import torch
    import tidegraph
    from tidegraph.native_loader import load_native
    from tidegraph.weighted.execution import eager_runner
    import tide_lab.core5 as application
    from tide_lab.train import runtime, synchronize
    from tide_lab.campaign_runtime import make_model, optimizer_for, synthetic_batch, memory, learning_rate
    from tide_lab.language import sequence_readout
    # Select the public execution option at the application adapter only.
    application.eager_runner = functools.partial(eager_runner, query_execution=a.query_execution,
                                               prefill_schedule=a.prefill_schedule)
    config = json.loads((source/CONFIG).read_text())
    config.update(device=a.device, threads=1 if a.device == "cpu" else 2)
    config["training"].update(batch=a.batch, length=a.length)
    config["core"].update(eager_implementation=a.client, placement_preset="cpu" if a.device == "cpu" else "mixed-c")
    if a.native_library:
        config["core"]["native_library"] = str(Path(a.native_library).resolve())
    record = dict(schema="tide-dense-query-cost-v1", execution="running", strict_numerical="not_checked",
        discrete="not_independently_checked", recovery="see separate consumer", learning="not_a_learning_study",
        downstream=DOWNSTREAM, downstream_archive_sha256=hashlib.sha256(archive).hexdigest(), resolved=config,
        query_execution=a.query_execution, prefill_schedule=a.prefill_schedule,
        installed_package=str(Path(tidegraph.__file__).resolve()),
        visibility=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"), affinity=sorted(os.sched_getaffinity(0)),
        phases=[], profile=a.profile, warmup_state=a.warmup_state,
        scope="One warmup, one measured step. Phase sum matches original diagnosis; complete_wall includes input creation/upload, finite check, cleanup and synchronization. Profile timing is separate.")
    write_json(out/"result.json", record)
    try:
        device = runtime(a.device, config["threads"], config["seed"])
        if a.client == "native":
            load_native(a.native_library, backend=device.type)
            manifest = Path(a.native_library)/"build-manifest.json"
            record["native_build"] = json.loads(manifest.read_text())
        start = time.monotonic()
        model = make_model(config, device); opt = optimizer_for(model, config)
        synchronize(device)
        record["construction_seconds"] = time.monotonic()-start
        record["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        initial_model = {n: t.detach().cpu().clone() for n, t in model.state_dict().items()} if a.warmup_state == "reset" else None
        initial_optimizer = copy.deepcopy(opt.state_dict()) if initial_model is not None else None
        def parameter_hash():
            h = hashlib.sha256()
            for name, value in model.state_dict().items():
                h.update(name.encode()+b"\0")
                h.update(value.detach().cpu().contiguous().numpy().tobytes())
            return h.hexdigest()
        record["initial_parameters_sha256"] = parameter_hash()

        def step(index):
            synchronize(device); wall = time.monotonic()
            batch = synthetic_batch(a.batch, a.length, 991+index, device)
            model.train()
            lr = learning_rate(config, (index+1)*a.batch*a.length)
            for group in opt.param_groups:
                group["lr"] = lr
            phases = {}
            def phase(name, fn):
                synchronize(device); begin = time.monotonic()
                with torch.profiler.record_function("task::"+name):
                    result = fn(); synchronize(device)
                phases[name] = time.monotonic()-begin
                return result
            phase("zero_grad", lambda: opt.zero_grad(set_to_none=True))
            features = phase("embedding", lambda: model.embedding(batch["input_ids"]))
            result = phase("core_forward", lambda: model.encoder.run_features(features, trace=False))
            hidden = phase("readout", lambda: sequence_readout(model.encoder, result, features))
            loss = phase("head_loss", lambda: torch.nn.functional.cross_entropy(model.head(hidden).flatten(0, 1), batch["labels"].flatten()))
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite diagnostic loss")
            phase("backward", loss.backward)
            norm = phase("clip", lambda: torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True))
            phase("optimizer", opt.step)
            row = dict(index=index, warmup=index == 0, phases_seconds=phases, phase_sum_seconds=sum(phases.values()),
                tokens=a.batch*a.length, loss=float(loss.detach()), gradient_norm=float(norm), work=dict(result.stats))
            del result, features, hidden, loss, norm, batch
            synchronize(device)
            row["complete_wall_seconds"] = time.monotonic()-wall
            row["memory"] = memory(device)
            return row

        for i in range(2):
            if i == 1:
                if initial_model is not None:
                    model.load_state_dict(initial_model); opt.load_state_dict(initial_optimizer)
                    opt.zero_grad(set_to_none=True)
                record["measured_parameters_sha256"] = parameter_hash()
            row = step(i); record["phases"].append(row)
            write_json(out/"result.json", record); print(json.dumps(row), flush=True)
        if a.profile != "none":
            if a.client == "native":
                import _tide_native as core
                core.reset_work(True)
            if a.profile == "cpu":
                profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True)
            else:
                import torch_npu
                profiler = torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU], record_shapes=True,
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(out/"npu-profile"), analyse_flag=True))
            with profiler:
                record["instrumented_step"] = step(2)
            if a.client == "native":
                record["instrumented_packing"] = dict(core.work_metrics())
                core.reset_work(False)
            if a.profile == "cpu":
                write_json(out/"operators.json", [dict(operator=e.key, count=e.count, input_shapes=e.input_shapes,
                    self_cpu_ms=e.self_cpu_time_total/1000) for e in profiler.key_averages(group_by_input_shape=True)])
            record["profile_state"] = "completed"
        record["execution"] = "completed"
    except BaseException as error:
        record.update(execution="failed", error=repr(error)); raise
    finally:
        write_json(out/"result.json", record)


if __name__ == "__main__":
    main()
