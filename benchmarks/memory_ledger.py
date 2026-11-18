"""Intrusive memory-lifetime ledger, never a normal-throughput measurement.

Reports unique live saved-for-backward storage, persistent owners, private state,
outputs and allocator snapshots separately. Shared views are not double counted.
Unclassified allocator memory is not relabelled as measured operator scratch.
"""
import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import resource
import subprocess

import torch

from tide import ModelConfig, TideModel, detach_state
from tide.precision import precision_context, precision_policy
from benchmarks.baselines import TransformerBaseline, parameter_count
from benchmarks.continuous import _tensors, forward_state
from benchmarks.references import file_identity, load_reference
from benchmarks.run import _device_memory, _loss, check_memory, synchronize
from benchmarks.statistics import RuntimeAudit
from benchmarks.workloads import activity_fixture, paired_input, parameter_hash, tensor_hash
from scripts.job import atomic


def storage_key(tensor):
    storage = tensor.untyped_storage()
    return str(tensor.device), storage.data_ptr()


def storage_inventory(value):
    tensors = list(_tensors(value))
    stores = {}
    for tensor in tensors:
        if tensor.untyped_storage().nbytes():
            key = storage_key(tensor)
            stores[key] = max(stores.get(key, 0), tensor.untyped_storage().nbytes())
    return stores, dict(tensor_count=len(tensors), payload_bytes=sum(t.numel()*t.element_size() for t in tensors),
                       unique_storage_count=len(stores), unique_storage_bytes=sum(stores.values()),
                       dtypes=sorted({str(t.dtype) for t in tensors}))


class _Saved:
    def __init__(self, tensor, owner):
        # A detached alias avoids retaining its grad_fn through the pack hook.
        # Storage stays alive until autograd releases this packed object.
        self.tensor, self.owner = tensor.detach(), owner
        self.key = storage_key(tensor)
        self.payload = tensor.numel()*tensor.element_size()
        owner._retain(self.key, tensor.untyped_storage().nbytes(), self.payload)

    def __del__(self):
        self.owner._release(self.key, self.payload)


class SavedStorageAudit:
    """Track simultaneous retained storage; unpack does not imply release."""
    def __init__(self, existing=()):
        self.existing = set(storage_inventory(existing)[0])
        self.live, self.counts = {}, Counter()
        self.pack_calls = self.unpack_calls = self.payload_events = 0
        self.live_payload = self.peak_payload = self.peak_unique = self.peak_additional = 0
        self.live_unique = self.live_additional = 0
        self.shapes = Counter()

    def _retain(self, key, size, payload):
        self.counts[key] += 1
        old = self.live.get(key, 0)
        self.live[key] = max(old, size)
        self.live_unique += self.live[key]-old
        if key not in self.existing:
            self.live_additional += self.live[key]-old
        self.live_payload += payload
        self.peak_payload = max(self.peak_payload, self.live_payload)
        self.peak_unique = max(self.peak_unique, self.live_unique)
        self.peak_additional = max(self.peak_additional, self.live_additional)

    def _release(self, key, payload):
        self.live_payload -= payload
        self.counts[key] -= 1
        if not self.counts[key]:
            self.live_unique -= self.live[key]
            if key not in self.existing:
                self.live_additional -= self.live[key]
            del self.live[key], self.counts[key]

    def pack(self, tensor):
        self.pack_calls += 1
        self.payload_events += tensor.numel()*tensor.element_size()
        self.shapes[(str(tensor.device), str(tensor.dtype), tuple(tensor.shape))] += 1
        return _Saved(tensor, self)

    def unpack(self, saved):
        self.unpack_calls += 1
        return saved.tensor

    def __enter__(self):
        self.context = torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack)
        self.context.__enter__()
        return self

    def __exit__(self, *args):
        return self.context.__exit__(*args)

    def report(self):
        return dict(pack_calls=self.pack_calls, unpack_calls=self.unpack_calls,
                    saved_payload_event_bytes=self.payload_events,
                    peak_live_saved_payload_bytes=self.peak_payload,
                    peak_live_unique_storage_bytes=self.peak_unique,
                    peak_live_additional_storage_bytes=self.peak_additional,
                    current_live_unique_storage_bytes=sum(self.live.values()),
                    current_live_unique_storage_count=len(self.live),
                    additional_scope="excludes storage already owned by parameters/slots/input/incoming state at the segment boundary",
                    lifetime="unique simultaneous storage held by packed objects; release on object destruction, not on unpack; repeated save payload counted separately",
                    hook_scope="outer PyTorch saved_tensors_hooks; nested hook overrides/custom opaque backend allocations may not be visible",
                    top_saved_shapes=[dict(device=device, dtype=dtype, shape=list(shape), save_calls=count)
                                      for (device, dtype, shape), count in self.shapes.most_common(32)])


def optimizer_tensors(optimizer):
    return [] if optimizer is None else [tensor for state in optimizer.state.values() for tensor in state.values()
                                         if isinstance(tensor, torch.Tensor)]


def snapshot(model, optimizer, inputs, state, values, saved, device):
    groups = dict(parameters=list(model.parameters()), gradients=[p.grad for p in model.parameters() if p.grad is not None],
                  optimizer=optimizer_tensors(optimizer), private_state=state, inputs=inputs, outputs=values)
    categories, all_stores, by_device = {}, {}, {}
    for name, value in groups.items():
        stores, row = storage_inventory(value)
        row["exclusive_unique_storage_bytes"] = sum(size for key, size in stores.items() if key not in all_stores)
        categories[name] = row
        all_stores.update(stores)
    stores = dict(saved.live) if saved is not None else {}
    categories["saved_for_backward"] = dict(unique_storage_count=len(stores), unique_storage_bytes=sum(stores.values()),
                                             exclusive_unique_storage_bytes=sum(size for key, size in stores.items() if key not in all_stores))
    all_stores.update(stores)
    for (name, _), size in all_stores.items():
        by_device[name] = by_device.get(name, 0)+size
    memory = _device_memory(device)
    return dict(categories=categories, unique_storage_union_bytes=sum(all_stores.values()),
                unique_storage_union_by_device=by_device,
                ownership_priority=list(categories), allocator=memory,
                live_allocator_minus_visible_union_bytes=(None if memory is None else memory["allocated_bytes"]-by_device.get(str(device), 0)),
                unclassified_scope="same-instant live allocator minus visible storage union; includes unobserved tensors and backend/allocator effects, not a scratch-only measurement",
                process_lifetime_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def forward_window(model, inputs, state=None, *, chunk=None, decode=0):
    prefill = inputs.shape[1]-decode
    if prefill < 1 or chunk is not None and chunk < 1:
        raise ValueError("positive prefill/chunk required")
    chunk = prefill if chunk is None else chunk
    schedule = [(start, min(prefill, start+chunk)) for start in range(0, prefill, chunk)]
    schedule += [(start, start+1) for start in range(prefill, inputs.shape[1])]
    values = []
    for start, stop in schedule:
        value, state, raw = forward_state(model, inputs[:, start:stop], state)
        values.append(value)
        del raw
    if isinstance(model, TideModel) and not model.settle:
        flushed = model.flush(state)
        values.append(flushed.readout())
        state = flushed.state
    return values, state


def diagnose(model, inputs, *, mode, precision, optimizer=None, state=None, chunk=None, decode=0,
             device_budget_gib=None):
    if (optimizer is not None) != (mode == "train"):
        raise ValueError("train requires optimizer; other modes omit it")
    device = inputs.device
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    existing = [list(model.parameters()), optimizer_tensors(optimizer), inputs, state]
    saved = SavedStorageAudit(existing)
    phases = {"boundary": snapshot(model, optimizer, inputs, state, [], None, device)}
    _device_memory(device, reset=True)
    # WorkAudit copies support to CPU and may re-enter regions during checkpoint
    # recomputation. It belongs to a separate semantic observation, not this
    # storage/runtime pass: its allocations and repeated counts are intrusive.
    with saved, RuntimeAudit() as runtime:
        with torch.set_grad_enabled(mode != "inference"), precision_context(device, precision):
            values, state = forward_window(model, inputs, state, chunk=chunk, decode=decode)
            loss = _loss(values) if optimizer is not None else None
        synchronize(device)
        check_memory(device, device_budget_gib)
        phases["forward"] = snapshot(model, optimizer, inputs, state, values, saved, device)
        if optimizer is not None:
            loss.backward()
            synchronize(device)
            check_memory(device, device_budget_gib)
            phases["backward"] = snapshot(model, optimizer, inputs, state, values, saved, device)
            optimizer.step()
            synchronize(device)
            check_memory(device, device_budget_gib)
            phases["update"] = snapshot(model, optimizer, inputs, state, values, saved, device)
    finite = all(bool(torch.isfinite(value).all()) for value in values)
    finite &= all(bool(torch.isfinite(p).all()) and (p.grad is None or bool(torch.isfinite(p.grad).all())) for p in model.parameters())
    finite &= all(bool(torch.isfinite(t).all()) for t in optimizer_tensors(optimizer))
    finite &= all(bool(torch.isfinite(t).all()) for t in _tensors(state))
    if not finite:
        raise ArithmeticError("nonfinite diagnostic output/parameters/gradients/optimizer/private state")
    if precision == "bf16" and not runtime.report()["bf16_matrix_observed"]:
        raise ArithmeticError("BF16 diagnostic observed no BF16 matrix output")
    state = detach_state(state)
    loss_value = None if loss is None else float(loss.detach())
    del values, loss
    gc.collect()
    synchronize(device)
    phases["released_outputs_and_graph"] = snapshot(model, optimizer, inputs, state, [], saved, device)
    runtime_report = runtime.report()
    covered = "forward/readout/loss/backward/optimizer" if optimizer is not None else "forward/readout"
    runtime_report["instrumentation"] = (f"intrusive {mode} diagnostic covering {covered}; with saved_tensors_hooks, "
                                         "without semantic WorkAudit hooks; not normal throughput timing")
    return state, dict(phases=phases,
                       phase_allocator_peak_scope="boundary is sampled before reset and includes prior work/warmup; "
                                                  "the reset follows boundary, so later phase peaks are cumulative "
                                                  "within this diagnostic segment, not per-phase increments",
                       saved_for_backward=saved.report(), runtime=runtime_report,
                       actual_work=None,
                       work_measurement="not observed in this ledger; use a separate semantic WorkAudit pass/reference",
                       input_positions=inputs.shape[0]*inputs.shape[1],
                       loss=loss_value, finite=True,
                       state_position=state.position, measurement="intrusive diagnostic; no normal timing/throughput result")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--subject", choices=("tide", "transformer_A", "transformer_B", "transformer_C", "transformer_D"), default="tide")
    parser.add_argument("--calibration-record")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--mode", choices=("inference", "grad_forward", "train"), default="train")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--prefill-chunk", type=int)
    parser.add_argument("--decode", type=int, default=0)
    parser.add_argument("--continue-state", action="store_true")
    parser.add_argument("--segments", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1107)
    parser.add_argument("--activity", choices=("natural", "sparse03", "sparse10", "high"), default="natural")
    parser.add_argument("--optimizer-foreach", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device-budget-gib", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.batch, args.length, args.segments, args.threads) < 1 or min(args.decode, args.warmup) < 0:
        parser.error("positive dimensions/segments/threads, nonnegative decode/warmup required")
    if args.prefill_chunk is not None and args.prefill_chunk < 1:
        parser.error("prefill chunk must be positive")
    if args.device_budget_gib is not None and not 0 < args.device_budget_gib < float("inf"):
        parser.error("device budget must be finite and positive")
    if args.device != "cpu" and args.device_budget_gib is None:
        parser.error("device diagnostics require an explicit allocator budget and outer launcher RSS/time limits")
    if args.output.exists():
        parser.error("choose a new diagnostic identity")
    if args.subject != "tide" and args.calibration_record is None:
        parser.error("Transformer diagnostic requires its frozen calibration record")
    if args.device.startswith("npu"):
        import torch_npu
        torch.npu.set_device(args.device)
        torch.npu.matmul.allow_hf32 = False
        torch.npu.conv.allow_hf32 = False
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    config = ModelConfig.from_dict(json.loads(Path(args.config).read_text()))
    source = os.environ.get("TIDE_SOURCE_COMMIT") or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    record = dict(schema="tide-memory-ledger-v1", source=source, state="preparing",
                  harness_source=os.environ.get("TIDE_HARNESS_COMMIT", source),
                  arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  config_file=file_identity(args.config), config=config.to_dict(), precision=precision_policy(device, args.precision),
                  physical_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"), cpu_affinity=sorted(os.sched_getaffinity(0)),
                  working_tree_dirty=(None if os.environ.get("TIDE_SOURCE_COMMIT") else bool(subprocess.check_output(
                      ["git", "status", "--porcelain", "--", "src/tide", "benchmarks"], text=True).strip())),
                  resource_identity=os.environ.get("TIDE_RESOURCE_IDENTITY", "unspecified"),
                  scope="intrusive lifetime/storage diagnostic; no throughput or numerical-equivalence qualification",
                  warmup_boundary="zero state each warmup; inference-only for continued-state diagnostics, otherwise selected mode",
                  state_boundary="fresh state per segment unless --continue-state; detach between segments; complete Timed flush each segment",
                  observations=[])
    atomic(args.output, record)
    try:
        torch.manual_seed(args.seed)
        if args.subject == "tide":
            model = TideModel(config)
            record["activity_fixture"] = activity_fixture(model, args.activity)
            width = config.input_dim or config.d
        else:
            args.no_flush, args.modes = False, [args.mode]
            reference = load_reference(args.calibration_record, config, args, same_shape=False)
            match = reference["joint_matches_by_mode"][args.mode][args.subject[-1]]
            record.update(calibration_record=file_identity(args.calibration_record), joint_match=match)
            model = TransformerBaseline(**match["constructor"])
            width = model.d
        record.update(parameters=parameter_count(model), initial_parameter_sha256=parameter_hash(model))
        x_cpu = paired_input(args.batch, args.length+args.decode, width, args.seed+1)
        record.update(input_sha256=tensor_hash(x_cpu), input_shape=list(x_cpu.shape))
        model = model.to(device).train(args.mode != "inference")
        inputs = x_cpu.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=args.optimizer_foreach) if args.mode == "train" else None
        record["scratch"] = dict(measured_operator_scratch_bytes=None,
                                 foreach_planning_allowance_bytes=4*record["parameters"] if args.mode == "train" and args.optimizer_foreach else 0,
                                 interpretation="allowance is not a measured allocation; saved storage, state and phase peaks overlap and must not be added")
        check_memory(device, args.device_budget_gib)
        record.update(state="running", last_stage="warmup")
        atomic(args.output, record)
        for _ in range(args.warmup):
            warm_mode = "inference" if args.continue_state else args.mode
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(warm_mode != "inference"), precision_context(device, args.precision):
                values, warm_state = forward_window(model, inputs, chunk=args.prefill_chunk, decode=args.decode)
                loss = _loss(values) if warm_mode == "train" else None
            if loss is not None:
                loss.backward()
                optimizer.step()
            del values, warm_state, loss
            synchronize(device)
            check_memory(device, args.device_budget_gib)
        record["post_warmup_parameter_sha256"] = parameter_hash(model)
        state = None
        for index in range(args.segments):
            record["last_stage"] = f"diagnostic_segment_{index+1}"
            atomic(args.output, record)
            state, row = diagnose(model, inputs, mode=args.mode, precision=args.precision, optimizer=optimizer,
                                  state=state if args.continue_state else None, chunk=args.prefill_chunk, decode=args.decode,
                                  device_budget_gib=args.device_budget_gib)
            row["segment"] = index+1
            record["observations"].append(row)
            atomic(args.output, record)
        record["state"] = "completed_finite"
    except Exception as error:
        record.update(state="failed", error=repr(error))
        import traceback
        traceback.print_exc()
    atomic(args.output, record)
    print(json.dumps(dict(output=str(args.output), state=record["state"], segments=len(record["observations"]))))
    return int(record["state"] != "completed_finite")


if __name__ == "__main__":
    raise SystemExit(main())
