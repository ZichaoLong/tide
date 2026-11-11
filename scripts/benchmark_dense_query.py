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
from fc_comparison_diagnostics import BOUNDARIES, boundary_scope, boundary_calls

DOWNSTREAM = "0439efd2af31cf10b07ce9e6423b0195f7449968"
CONFIG = "configs/studies/language/v6/scale-fc16-kv-quad.json"


def task_source(repository, source):
    """Read only the pinned downstream Git objects, never its working files."""
    source.mkdir()
    archive = subprocess.check_output(["git", "-C", str(repository), "archive", DOWNSTREAM, "tide_lab", CONFIG])
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(source, filter="data")
    sys.path.insert(0, str(source))
    return json.loads((source/CONFIG).read_text()), hashlib.sha256(archive).hexdigest()


def geometry(config, shape):
    if shape == "fc4":
        config["core"]["recipe"].update(depth=2, region_width=2)
        config.pop("planned_inventory", None)
    elif shape != "fc16":
        raise ValueError("unknown FC comparison geometry")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--downstream-repo", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "npu:0"), required=True)
    p.add_argument("--client", choices=("python", "native"), default="native")
    p.add_argument("--baseline", choices=("graph", "fc-tensor", "transformer-16", "transformer-4", "transformer-2"), default="graph")
    p.add_argument("--fc-shape", choices=("fc16", "fc4"), default="fc16",
                   help="fc16=4 regions x4 nodes; fc4=2x2 with the same local dimensions and strength ratio")
    p.add_argument("--phase-fences", choices=("phase", "step"), default="phase",
                   help="phase preserves historical per-phase device fences; step fences only the whole request/update")
    p.add_argument("--diagnostic-boundary", choices=BOUNDARIES, default="public",
                   help="Python graph only: omit selected independent-root wrappers for task-projection diagnosis, never a public implementation")
    p.add_argument("--formula-attention", choices=("manual", "sdpa", "sdpa4d"), default="manual",
                   help="fc-tensor only: isolate the Full Attention operator; default retains the independent explicit formula")
    p.add_argument("--mode", choices=("training", "inference", "nograd", "grad-forward", "backward"), default="training",
                   help="nograd/grad-forward/backward/training use the same full-sequence head/loss; inference measures last-position first-token return")
    p.add_argument("--threads", type=int, help="CPU intra-op threads; defaults retain historical CPU=1/NPU=2")
    p.add_argument("--autograd-census", action="store_true", help="separate instrumented pass; its times are not formal throughput")
    p.add_argument("--query-execution", choices=("auto", "legacy", "dense-v1", "packed-v2"), default="auto")
    p.add_argument("--prefill-schedule", choices=("auto", "greedy", "window"), default="auto")
    p.add_argument("--trace-records", choices=("complete", "minimal"), default="complete",
                   help="native historical costs retain complete event records; minimal is a separate return contract")
    p.add_argument("--native-library")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--length", type=int, default=128)
    p.add_argument("--profile", choices=("none", "stages", "cpu", "npu"), default="none")
    p.add_argument("--profile-shapes", action="store_true", help="retain profiler shapes; increases diagnostic memory")
    p.add_argument("--profile-output", type=Path, help="optional scratch directory for large NPU traces; retain reviewed summaries durably")
    p.add_argument("--warmup-state", choices=("reset", "continue"), default="reset",
                   help="reset restores initial weights/optimizer after warmup for controlled comparisons")
    a = p.parse_args()
    if a.baseline == "graph" and a.client == "native" and not a.native_library:
        p.error("native requires an explicit matching --native-library build directory")
    if a.profile == "stages" and (a.client != "native" or a.baseline != "graph"):
        p.error("native stage counters require --client native")
    if a.diagnostic_boundary != "public" and (a.baseline != "graph" or a.client != "python"):
        p.error("diagnostic boundary trials require the Python graph client")
    if a.formula_attention != "manual" and a.baseline != "fc-tensor":
        p.error("formula Attention trial requires --baseline fc-tensor")
    if not (1 <= a.batch <= 8 and 1 <= a.length <= 128):
        p.error("this bounded probe requires 1<=B<=8, 1<=L<=128")
    if a.threads is not None and a.threads < 1:
        p.error("--threads must be positive")
    out = a.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    config, archive_hash = task_source(a.downstream_repo, out/"downstream-source")
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
                                               prefill_schedule=a.prefill_schedule,
                                               trace=a.trace_records == "complete")
    geometry(config, a.fc_shape)
    config.update(device=a.device, threads=a.threads if a.threads is not None else 1 if a.device == "cpu" else 2)
    config["training"].update(batch=a.batch, length=a.length)
    config["core"].update(eager_implementation=a.client, placement_preset="cpu" if a.device == "cpu" else "mixed-c")
    if a.native_library:
        config["core"]["native_library"] = str(Path(a.native_library).resolve())
    from source_identity import source_state
    foundation_source, foundation_dirty = source_state(Path(__file__).resolve().parents[1])
    record = dict(foundation_source=foundation_source, foundation_dirty=foundation_dirty, schema="tide-dense-query-cost-v1", execution="running", strict_numerical="not_checked",
        discrete="not_independently_checked", recovery="see separate consumer", learning="not_a_learning_study",
        downstream=DOWNSTREAM, downstream_archive_sha256=archive_hash, resolved=config,
        query_execution=a.query_execution, prefill_schedule=a.prefill_schedule, trace_records=a.trace_records,
        baseline=a.baseline, mode=a.mode, fc_shape=a.fc_shape, phase_fences=a.phase_fences,
        diagnostic_boundary=a.diagnostic_boundary,
        formula_attention=a.formula_attention,
        boundary_scope="Public obligations" if a.diagnostic_boundary == "public" else
            "Diagnostic task projection: selected independent intermediate-root VJP wrappers omitted; not qualified for general graph roots",
        requested_threads=a.threads, autograd_census_requested=a.autograd_census,
        installed_package=str(Path(tidegraph.__file__).resolve()),
        visibility=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"), affinity=sorted(os.sched_getaffinity(0)),
        phases=[], profile=a.profile, warmup_state=a.warmup_state,
        scope="One warmup, one measured step. Phase sum matches original diagnosis; complete_wall includes input creation/upload, finite check, cleanup and synchronization. Profile timing is separate.")
    write_json(out/"result.json", record)
    try:
        device = runtime(a.device, config["threads"], config["seed"])
        from layer_tensor_baseline import numerics
        record["precision"] = numerics(device, "vendor-default")
        full_sequence = a.mode != "inference"
        torch.set_grad_enabled(a.mode in {"training", "grad-forward", "backward"})
        core = None
        if a.baseline == "graph" and a.client == "native":
            core = load_native(a.native_library, backend=device.type)
            manifest = Path(a.native_library)/"build-manifest.json"
            record["native_build"] = json.loads(manifest.read_text())
            if a.autograd_census and not all(hasattr(core, n) for n in (
                    "reset_autograd_census", "autograd_census_metrics", "register_autograd_roots")):
                raise RuntimeError("native autograd census requires a matching build with explicit private-graph instrumentation")
        record["effective_threads"] = torch.get_num_threads()
        record["effective_interop_threads"] = torch.get_num_interop_threads()
        start = time.monotonic()
        model = make_model(config, device)
        if a.baseline.startswith("transformer-"):
            from fc_tensor_baseline import TransformerLayers
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(config["seed"])
                model.encoder = TransformerLayers(depth=int(a.baseline.split('-')[-1])).to(device)
        opt = optimizer_for(model, config)
        record["comparison_scope"] = ("Original graph task" if a.baseline == "graph" else
            "Same core-5 parameters/input/head/loss; independent FC output/terminal-state projection, no complete trace or None-root equivalence" if a.baseline == "fc-tensor" else
            "Different function: serial pre-RMS/SDPA/SwiGLU Transformer, d128/head2/FFN512; same input/head/loss/optimizer, actual parameter count recorded")
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

        @boundary_scope(a.diagnostic_boundary)
        def step(index, census=None):
            synchronize(device); wall = time.monotonic()
            batch = synthetic_batch(a.batch, a.length, 991+index, device)
            model.train(full_sequence)
            lr = learning_rate(config, (index+1)*a.batch*a.length)
            for group in opt.param_groups:
                group["lr"] = lr
            phases = {}
            def phase(name, fn):
                if a.phase_fences == "phase":
                    synchronize(device)
                begin = time.monotonic()
                with torch.profiler.record_function("task::"+name):
                    result = fn()
                    if a.phase_fences == "phase":
                        synchronize(device)
                phases[name] = time.monotonic()-begin
                return result
            if a.mode in {"training", "backward"}:
                phase("zero_grad", lambda: opt.zero_grad(set_to_none=True))
            features = phase("embedding", lambda: model.embedding(batch["input_ids"]))
            if a.baseline == "graph":
                result = phase("core_forward", lambda: model.encoder.run_features(features, trace=a.trace_records == "complete"))
                hidden = phase("readout", lambda: sequence_readout(model.encoder, result, features)
                               if full_sequence else model.encoder.project(result, features))
                work = dict(result.stats)
            elif a.baseline == "fc-tensor":
                from fc_tensor_baseline import run
                result = phase("core_forward", lambda: run(model.encoder.binding, model.encoder.spec, features,
                                                          full_attention=a.formula_attention))
                hidden = result["hidden"] if full_sequence else result["hidden"][:, -1]
                work = result["stats"]
            else:
                result = phase("core_forward", lambda: model.encoder(features))
                hidden = result if full_sequence else result[:, -1]
                work = dict(block_positions=a.batch*a.length*len(model.encoder.blocks))
            norm, loss = None, None
            if full_sequence:
                loss = phase("head_loss", lambda: torch.nn.functional.cross_entropy(model.head(hidden).flatten(0, 1), batch["labels"].flatten()))
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("nonfinite diagnostic loss")
            else:
                logits = phase("head", lambda: model.head(hidden))
                token = phase("first_token", lambda: logits.argmax(-1))
                if not bool(torch.isfinite(logits).all()):
                    raise FloatingPointError("nonfinite first-token logits")
            if census is not None:
                from tidegraph.autograd_census import tensor_roots
                def register():
                    roots = list(tensor_roots((result, features, hidden, loss if full_sequence else logits)))
                    census.register(roots, label="outer/returned-and-task")
                    if core is not None:
                        core.register_autograd_roots("outer/returned-and-task", roots)
                phase("census_register", register)
            if a.mode in {"training", "backward"}:
                phase("backward", loss.backward)
            if a.mode == "training":
                norm = phase("clip", lambda: torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True))
                phase("optimizer", opt.step)
            row = dict(index=index, warmup=index == 0, phases_seconds=phases, phase_sum_seconds=sum(phases.values()),
                diagnostic_boundary_calls=boundary_calls(),
                phase_time_scope="synchronized stage intervals" if a.phase_fences == "phase" else
                    "host enqueue/calling-thread intervals; device work may cross phases; only complete_wall_seconds is synchronized",
                tokens=a.batch*a.length, loss=None if loss is None else float(loss.detach()),
                gradient_norm=None if norm is None else float(norm), work=work)
            if census is not None:
                row["autograd"] = dict(python=census.snapshot(), native=None if core is None else dict(core.autograd_census_metrics()),
                    scope="All returned tensor fields plus task roots, and explicitly instrumented private graphs. Python/native node universes overlap: never add their totals. Native saved tensors are explicit saves only, not full ATen saved memory. Observed after backward/optimizer when requested, before returned-root cleanup.")
            if a.mode == "inference":
                row["first_tokens"] = token.cpu().tolist()
                row["ttft_scope"] = "Warm loaded local batch; input creation/upload, prompt prefill, last-position head, greedy first-token selection, finite check, host return; no queue, tokenizer or subsequent decode"
                row["first_token_seconds"] = time.monotonic()-wall
                del logits, token
            del result, features, hidden, loss, norm, batch
            synchronize(device)
            row["complete_wall_seconds"] = time.monotonic()-wall
            row["diagnostic_only"] = census is not None
            if census is None:
                row["input_positions_per_second"] = a.batch*a.length/row["complete_wall_seconds"]
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
            if initial_model is not None:
                model.load_state_dict(initial_model); opt.load_state_dict(initial_optimizer)
                opt.zero_grad(set_to_none=True)
            record["profile_parameters_sha256"] = parameter_hash()
            if core is not None:
                core.reset_work(True)
                core.reset_stage_profile(True)
            if a.profile == "stages":
                from contextlib import nullcontext
                profiler = nullcontext()
            elif a.profile == "cpu":
                profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=a.profile_shapes)
            else:
                import torch_npu
                profile_output = (a.profile_output or out/"npu-profile").resolve()
                record["profile_output"] = str(profile_output)
                profiler = torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU], record_shapes=a.profile_shapes,
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(profile_output), analyse_flag=True))
            from fc_comparison_diagnostics import formula_ranges
            with profiler, formula_ranges():
                record["instrumented_step"] = step(1)
            if core is not None:
                record["instrumented_packing"] = dict(core.work_metrics())
                record["instrumented_stages"] = dict(core.stage_profile_metrics())
                record["stage_scope"] = "Exclusive calling-thread elapsed time, includes incidental waits; not device kernel time or additive multiworker wall time. Reset and read only with workers idle."
                core.reset_work(False)
                core.reset_stage_profile(False)
            if a.profile == "cpu":
                write_json(out/"operators.json", [dict(operator=e.key, count=e.count, input_shapes=e.input_shapes,
                    self_cpu_ms=e.self_cpu_time_total/1000, total_cpu_ms=e.cpu_time_total/1000)
                    for e in profiler.key_averages(group_by_input_shape=a.profile_shapes)])
                from fc_comparison_diagnostics import stage_operators
                write_json(out/"stage-operators.json", stage_operators(profiler))
            record["profile_state"] = "completed"
        if a.autograd_census:
            from tidegraph.autograd_census import Census
            if initial_model is not None:
                model.load_state_dict(initial_model); opt.load_state_dict(initial_optimizer)
                opt.zero_grad(set_to_none=True)
            record["census_parameters_sha256"] = parameter_hash()
            if core is not None:
                core.reset_autograd_census(True)
            try:
                with Census() as census:
                    diagnostic = step(1, census)
                write_json(out/"autograd-census.json", diagnostic)
                record["autograd_census"] = "autograd-census.json"
                record["census_scope"] = "Separate pass after timing, restored initial parameters/optimizer when warmup_state=reset; instrumented phase times are not formal throughput."
            finally:
                if core is not None:
                    core.reset_autograd_census(False)
        record["execution"] = "completed"
    except BaseException as error:
        record.update(execution="failed", error=repr(error)); raise
    finally:
        write_json(out/"result.json", record)


if __name__ == "__main__":
    main()
