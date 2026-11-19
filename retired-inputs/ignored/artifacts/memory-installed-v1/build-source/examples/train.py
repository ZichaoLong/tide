"""Public-API training/continuation example with explicit, reviewable settings.

Synthetic inputs and a scalar mechanism objective are not a downstream recipe.
The example, including --resume, imports no benchmark or repository internals.
"""
import argparse
from dataclasses import asdict, fields, is_dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import re


PRESETS = Path(__file__).resolve().parents[1]/"configs/recommended"
EXECUTION_DEFAULTS = dict(mode="train", device="cpu", precision="fp32", batch=2, length=8,
                          steps=3, threads=1, learning_rate=1e-4, foreach=True, seed=17)
MODEL_OPTIONS = ("memory", "region_batch_size", "checkpoint_regions", "checkpoint_reception",
                 "ssd_summary_backend", "ssd_scan_backend", "attention_backend", "local_attention_backend",
                 "aggregation", "reception_qkv_backend", "reception_mask_padding", "parameter_indexing",
                 "kv_width", "ffn_width", "memory_sharing")
UPDATE_TOLERANCE = dict(atol=1e-6, rtol=2e-3)


def emit(kind, **record):
    print(json.dumps(dict(kind=kind, **record), ensure_ascii=False), flush=True)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--preset", help="name under configs/recommended or an explicit preset JSON path")
    result.add_argument("--config", type=Path, help="replace the preset's model with a ModelConfig JSON")
    result.add_argument("--list-presets", action="store_true")
    result.add_argument("--dry-run", action="store_true", help="resolve settings without model weights or device initialization")
    result.add_argument("--mode", choices=("train", "grad_forward", "inference"))
    result.add_argument("--device", help="explicit CPU/CUDA/NPU device; never auto-selected")
    result.add_argument("--precision", choices=("fp32", "bf16"))
    for name in ("batch", "length", "steps", "threads", "seed"):
        result.add_argument("--"+name, type=int)
    result.add_argument("--lr", type=float, dest="learning_rate")
    result.add_argument("--foreach", action=argparse.BooleanOptionalAction, default=None)
    result.add_argument("--memory", choices=("K", "S", "X", "X_reverse"))
    result.add_argument("--kv-width", type=int, help="total internal Q/K/V width, not GQA")
    result.add_argument("--ffn-width", type=int, help="Full FFN hidden width; default 4*d")
    result.add_argument("--memory-sharing", choices=("region_shared",),
                        help="share only memory bases; preserve other module ownership")
    result.add_argument("--region-batch-size", type=int)
    result.add_argument("--checkpoint-regions", action=argparse.BooleanOptionalAction, default=None)
    result.add_argument("--checkpoint-reception", action=argparse.BooleanOptionalAction, default=None)
    result.add_argument("--aggregation", choices=("source_residual", "incoming_light_64", "incoming_light_128", "incoming_wide"))
    result.add_argument("--reception-qkv-backend", choices=("separate", "fused"))
    result.add_argument("--reception-mask-padding", action=argparse.BooleanOptionalAction, default=None)
    result.add_argument("--ssd-summary-backend", choices=("block", "prefix"))
    result.add_argument("--ssd-scan-backend", choices=("block", "incremental", "doubling"))
    result.add_argument("--parameter-indexing", choices=("gather", "static_view"))
    for name in ("attention-backend", "local-attention-backend"):
        result.add_argument("--"+name, choices=("auto", "math", "sdpa", "tiled", "npu_fusion"))
    result.add_argument("--checkpoint", type=Path,
                        help="train only: save to a new path, then verify an actual paired restored update")
    result.add_argument("--resume", type=Path, help="restore trusted model/state/AdamW; RNG and data stream are not restored")
    return result


def read_json(path):
    content = path.read_bytes()
    return json.loads(content), dict(path=str(path.resolve()), sha256=hashlib.sha256(content).hexdigest())


def resolve(args):
    from tide import ModelConfig, build_topology
    name = args.preset or "default"
    candidate = Path(name)
    path = candidate if candidate.suffix == ".json" or candidate.parent != Path(".") else PRESETS/(name+".json")
    preset, preset_source = read_json(path)
    if not isinstance(preset, dict) or preset.get("schema") != "tide-example-preset-1":
        raise ValueError("preset must use schema tide-example-preset-1")
    if set(preset)-{"schema", "description", "model", "execution", "recommendation"}:
        raise ValueError("unknown preset field")
    execution = dict(EXECUTION_DEFAULTS)
    chosen = preset.get("execution", {})
    if not isinstance(chosen, dict) or set(chosen)-set(execution):
        raise ValueError("unknown or invalid preset execution setting")
    execution.update(chosen)
    execution.update({key:getattr(args, key) for key in execution if getattr(args, key) is not None})
    if execution["device"] is None:
        raise ValueError("this preset requires an explicit --device; select an admitted npu:N for NPU training")
    model_data, config_source = preset.get("model"), None
    if args.config is not None:
        model_data, config_source = read_json(args.config)
    config = ModelConfig.from_dict(model_data)
    config = replace(config, **{name:getattr(args, name) for name in MODEL_OPTIONS if getattr(args, name) is not None})
    graph = build_topology(config.topology)
    for key in ("batch", "length", "steps", "threads"):
        if type(execution[key]) is not int or execution[key] < 1:
            raise ValueError(key+" must be a positive integer")
    if type(execution["seed"]) is not int or not 0 <= execution["seed"] < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    if (type(execution["learning_rate"]) not in (int, float) or not math.isfinite(execution["learning_rate"])
            or execution["learning_rate"] <= 0):
        raise ValueError("learning_rate must be finite and positive")
    if type(execution["foreach"]) is not bool:
        raise ValueError("foreach must be boolean")
    if execution["mode"] not in ("train", "grad_forward", "inference"):
        raise ValueError("unknown execution mode")
    if execution["precision"] not in ("fp32", "bf16"):
        raise ValueError("precision must be fp32 or bf16")
    if not isinstance(execution["device"], str) or not re.fullmatch(r"(?:cpu|cuda|npu)(?::\d+)?", execution["device"]):
        raise ValueError("device must be an explicit cpu, cuda or npu device")
    if execution["mode"] != "train" and (args.checkpoint is not None or args.learning_rate is not None or args.foreach is not None):
        raise ValueError("--checkpoint, --lr and --[no-]foreach require --mode train")
    if args.checkpoint is not None and args.checkpoint.exists():
        raise ValueError("choose a new checkpoint path")
    if args.resume is not None and not args.resume.is_file():
        raise ValueError("resume checkpoint does not exist")
    return dict(config=config, execution=execution, graph=graph,
                source=dict(preset=preset_source, config=config_source,
                            explicit_overrides={key:getattr(args, key) for key in (*MODEL_OPTIONS, *execution)
                                                if getattr(args, key) is not None}),
                recommendation=preset.get("recommendation", {}),
                explicit_model=args.config is not None or args.preset is not None,
                model_overrides={key:getattr(args, key) for key in MODEL_OPTIONS if getattr(args, key) is not None})


def tree(value):
    """Public dataclass state -> ordinary containers for exact Torch comparisons."""
    if is_dataclass(value):
        return {field.name:tree(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {key:tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(tree(item) for item in value)
    if isinstance(value, list):
        return [tree(item) for item in value]
    return value


def finite(value):
    import torch
    value = tree(value)
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite(item) for item in value)
    return True


def window(model, x, state, precision):
    from tide.precision import precision_context
    with precision_context(x.device, precision):
        result = model(x, state)
        tail = model.flush(result.state) if model.config.topology.timing == "timed" else None
        objective = result.readout().float().square().mean()
        if tail is not None:
            objective = objective+tail.readout().float().square().mean()
    return result, tail, objective


def update(model, optimizer, x, state, precision):
    from tide import detach_state
    optimizer.zero_grad(set_to_none=True)
    result, tail, objective = window(model, x, state, precision)
    objective.backward()
    if not finite(objective) or not all(finite(parameter.grad) for parameter in model.parameters()):
        raise ArithmeticError("nonfinite objective or gradient")
    optimizer.step()
    following = detach_state((result if tail is None else tail).state)
    if not finite(model.state_dict()) or not finite(optimizer.state_dict()) or not finite(following):
        raise ArithmeticError("nonfinite updated parameter, optimizer or private state")
    return result, tail, objective, following


def verify_restore(path, model, optimizer, state, x, precision):
    import torch
    from tide import load_checkpoint
    restored, restored_state, slots = load_checkpoint(path, device=x.device)
    resumed = torch.optim.AdamW(restored.parameters())
    resumed.load_state_dict(slots)
    for actual, expected in ((restored.state_dict(), model.state_dict()), (tree(restored_state), tree(state)),
                             (resumed.state_dict(), optimizer.state_dict())):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, check_device=False)
    with torch.no_grad():
        expected = window(model, x, state, precision)
        actual = window(restored, x, restored_state, precision)
        torch.testing.assert_close(tree(actual), tree(expected), rtol=0, atol=0, check_device=False)
    del expected, actual
    states = []
    for current, current_optimizer, current_state in ((model, optimizer, state), (restored, resumed, restored_state)):
        *_, following = update(current, current_optimizer, x, current_state, precision)
        states.append(following)
    failures, maximum = [], 0.0
    for (name, actual), (_, expected) in zip(restored.named_parameters(), model.named_parameters()):
        delta = (actual-expected).detach().abs()
        largest = float(delta.max()); maximum = max(maximum, largest)
        if bool((delta > UPDATE_TOLERANCE["atol"]+UPDATE_TOLERANCE["rtol"]*expected.detach().abs()).any()):
            failures.append(dict(parameter=name, max_abs=largest))
    optimizer_difference = None
    try:
        torch.testing.assert_close(resumed.state_dict(), optimizer.state_dict(), **UPDATE_TOLERANCE, check_device=False)
    except AssertionError as error:
        optimizer_difference = str(error)
    torch.testing.assert_close(tree(states[1]), tree(states[0]), rtol=0, atol=0, check_device=False)
    report = dict(checkpoint=str(path), checkpoint_data="exact", next_forward="exact",
                  restored_state_position=states[1].position, paired_real_updates=2,
                  next_optimizer_update="finite_deviation" if failures or optimizer_difference else "within_threshold",
                  update_threshold=UPDATE_TOLERANCE, parameter_max_abs=maximum,
                  parameter_threshold_failures=failures, optimizer_difference=optimizer_difference,
                  saved_boundary="checkpoint remains before the two verification updates")
    return report, states[0]


def run(args, settings):
    import torch
    from tide import TideModel, detach_state, load_checkpoint, placement_report, save_checkpoint
    from tide.precision import precision_policy
    config, execution = settings["config"], dict(settings["execution"])
    device = execution["device"]
    if device.startswith("npu"):
        import torch_npu  # Explicit NPU selection only; module loading is not device admission.
        torch.npu.set_device(device)
        torch.npu.matmul.allow_hf32 = False
        torch.npu.conv.allow_hf32 = False
    torch.set_num_threads(execution["threads"])
    torch.manual_seed(execution["seed"])
    state, slots = None, None
    if args.resume is None:
        model = TideModel(config).to(device)
    else:
        model, state, slots = load_checkpoint(args.resume, device=device)
        try:
            placement_report(model)
        except ValueError:  # Public report rejects an ordinary single-device model.
            pass
        else:
            raise ValueError("this single-device example does not resume placed checkpoints; use the placement API")
        if (settings["explicit_model"] and model.config != config or
                any(getattr(model.config, key) != value for key, value in settings["model_overrides"].items())):
            raise ValueError("explicit model configuration differs from the checkpoint; resume preserves saved model options")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError("this FP32/BF16 example requires checkpoint parameters stored in FP32")
    model.train(execution["mode"] != "inference")
    optimizer = None
    if execution["mode"] == "train":
        optimizer = torch.optim.AdamW(model.parameters(), lr=execution["learning_rate"], foreach=execution["foreach"])
        if args.resume is not None:
            if (not isinstance(slots, dict) or len(slots.get("param_groups", [])) != 1 or
                    not {"betas", "eps", "weight_decay", "amsgrad"} <= slots["param_groups"][0].keys()):
                raise ValueError("training resume requires a saved single-group AdamW optimizer")
            optimizer.load_state_dict(slots)
            for key, option in (("lr", "learning_rate"), ("foreach", "foreach")):
                if getattr(args, option) is not None:
                    optimizer.param_groups[0][key] = getattr(args, option)
                execution[option] = optimizer.param_groups[0][key]
    emit("resolved_settings", model=model.config.to_dict(), topology=asdict(model.graph.config),
         clock=asdict(model.clock), execution=execution, source=settings["source"], recommendation=settings["recommendation"],
         precision=precision_policy(device, execution["precision"]), parameters=sum(p.numel() for p in model.parameters()),
         actual_parameter_devices=sorted({str(p.device) for p in model.parameters()}),
         actual_parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
         optimizer=None if optimizer is None else {k:v for k,v in optimizer.param_groups[0].items() if k != "params"},
         resume=None if args.resume is None else str(args.resume), state_position=0 if state is None else state.position,
         data="fresh synthetic inputs from this invocation's seed; no RNG/data-stream restore",
         boundary="each finite Timed window includes flush in the objective and advances the saved clock; detach between windows",
         purpose="mechanism example; grad_forward performs no backward/update; no downstream recipe or performance claim")
    generator = torch.Generator(device="cpu").manual_seed(execution["seed"])
    def next_input():
        return torch.randn(execution["batch"], execution["length"], model.config.input_dim or model.config.d,
                           generator=generator).to(device)
    for index in range(execution["steps"]):
        before = 0 if state is None else state.position
        context = torch.inference_mode() if execution["mode"] == "inference" else torch.enable_grad()
        with context:
            x = next_input()
            if optimizer is not None:
                result, tail, objective, state = update(model, optimizer, x, state, execution["precision"])
            else:
                result, tail, objective = window(model, x, state, execution["precision"])
                state = detach_state((result if tail is None else tail).state)
                if not finite((result, tail, objective)):
                    raise ArithmeticError("nonfinite forward result or private state")
        emit("step", index=index, mode=execution["mode"], objective=float(objective.detach()),
             state_before_position=before, state_position=state.position, flushed=tail is not None,
             forward_requires_grad=result.values.requires_grad, optimizer_updated=optimizer is not None)
        del result, tail, objective, x
    if args.checkpoint is not None:
        save_checkpoint(args.checkpoint, model, state=state, optimizer=optimizer)
        report, state = verify_restore(args.checkpoint, model, optimizer, state, next_input(), execution["precision"])
        emit("restore_check", **report)
    return dict(model=model, state=state, optimizer=optimizer, execution=execution)


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    try:
        if args.list_presets:
            presets = []
            for path in sorted(PRESETS.glob("*.json")):
                value, _ = read_json(path)
                presets.append(dict(name=path.stem, description=value.get("description"),
                                    execution=value.get("execution"), status=value.get("recommendation", {}).get("status")))
            emit("presets", directory=str(PRESETS), presets=presets)
            return 0
        settings = resolve(args)
        if args.dry_run:
            emit("planned_settings", model=settings["config"].to_dict() if args.resume is None else "checkpoint-authoritative; decoded only when run",
                 execution=settings["execution"], source=settings["source"], recommendation=settings["recommendation"],
                 device_initialized=False)
            return 0
        run(args, settings)
    except (OSError, ValueError, TypeError, KeyError) as error:
        cli.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
