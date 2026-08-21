"""Training and live-device probes for the v0 TIDE experiment."""

from __future__ import annotations

import json
import math
import os
import pathlib
import random
import time
from collections.abc import Mapping
from typing import Any

from .data import TokenSequenceDataset, deterministic_batches, load_data_manifest
from .manifest import atomic_write_json, base_manifest, model_identity, sha256_file


CHECKPOINT_SCHEMA = "tide-training-v1"


def _jsonable_metrics(metrics: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for layer_index, layer_metrics in metrics.items():
        converted: dict[str, Any] = {}
        for name, value in layer_metrics.items():
            cpu_value = value.detach().float().cpu()
            converted[name] = cpu_value.tolist() if cpu_value.ndim else float(cpu_value)
        result[str(layer_index)] = converted
    return result


def _atomic_torch_save(torch: Any, payload: dict[str, Any], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _to_cpu(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def _move_optimizer_state(optimizer: Any, device: Any) -> None:
    import torch

    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _capture_rng(runtime: Any) -> dict[str, Any]:
    torch = runtime.torch
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if runtime.info.backend == "cuda":
        state["accelerator"] = torch.cuda.get_rng_state(runtime.info.device_index).cpu()
    elif runtime.info.backend == "npu":
        state["accelerator"] = torch.npu.get_rng_state(runtime.info.device_index).cpu()
    return state


def _restore_rng(runtime: Any, state: Mapping[str, Any]) -> None:
    torch = runtime.torch
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if "accelerator" in state:
        if runtime.info.backend == "cuda":
            torch.cuda.set_rng_state(state["accelerator"], runtime.info.device_index)
        elif runtime.info.backend == "npu":
            torch.npu.set_rng_state(state["accelerator"], runtime.info.device_index)


def _learning_rate_scale(step: int, total_steps: int, warmup_ratio: float, minimum_ratio: float) -> float:
    warmup_steps = max(1, round(total_steps * warmup_ratio))
    if step <= warmup_steps:
        return step / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_ratio + (1.0 - minimum_ratio) * cosine


def _gradient_coverage(model: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for layer_index, wrapped in model.wrapped_layers.items():
        receiver_values = []
        for receiver in wrapped.receiver_group.ffns:
            gradient = receiver.down_proj.weight.grad
            receiver_values.append(None if gradient is None else float(gradient.float().norm().detach().cpu()))
        result[str(layer_index)] = receiver_values
    return result


def evaluate(
    *,
    model: Any,
    runtime: Any,
    dataset: TokenSequenceDataset,
    batch_size: int,
    max_tokens: int,
) -> dict[str, float]:
    torch = runtime.torch
    model.eval()
    total_loss = 0.0
    total_sequences = 0
    maximum_sequences = min(len(dataset), max_tokens // dataset.sequence_length)
    started = time.monotonic()
    with torch.no_grad():
        for start in range(0, maximum_sequences, batch_size):
            indices = range(start, min(start + batch_size, maximum_sequences))
            input_ids = torch.stack([dataset[index] for index in indices]).to(runtime.device)
            attention_mask = torch.ones_like(input_ids)
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            batch_sequences = input_ids.shape[0]
            total_loss += float(output.lm_loss.float().cpu()) * batch_sequences
            total_sequences += batch_sequences
    runtime.synchronize()
    mean_loss = total_loss / max(1, total_sequences)
    return {
        "loss": mean_loss,
        "perplexity": math.exp(min(20.0, mean_loss)),
        "sequences": float(total_sequences),
        "tokens": float(total_sequences * dataset.sequence_length),
        "seconds": time.monotonic() - started,
    }


def _save_checkpoint(
    *,
    model: Any,
    optimizer: Any,
    runtime: Any,
    output_dir: pathlib.Path,
    step: int,
    tokens: int,
    consumed_sequences: int,
    configuration: dict[str, Any],
) -> pathlib.Path:
    torch = runtime.torch
    path = output_dir / "checkpoints" / f"step-{step:06d}.pt"
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "configuration": configuration,
        "progress": {
            "step": step,
            "tokens": tokens,
            "consumed_sequences": consumed_sequences,
        },
        "model": _to_cpu(model.state_dict()),
        "optimizer": _to_cpu(optimizer.state_dict()),
        "rng": _capture_rng(runtime),
    }
    _atomic_torch_save(torch, payload, path)
    atomic_write_json(
        output_dir / "checkpoints" / "latest.json",
        {"schema": CHECKPOINT_SCHEMA, "path": path.name, "step": step, "tokens": tokens},
    )
    return path


def run_training(runtime: Any, arguments: dict[str, Any]) -> dict[str, Any]:
    torch = runtime.torch
    from .qwen3 import load_qwen3_model

    output_dir = pathlib.Path(arguments["output_dir"]).resolve()
    resume_path = arguments.get("resume")
    init_path = arguments.get("init_from")
    if resume_path:
        if not pathlib.Path(resume_path).is_file():
            raise FileNotFoundError(f"resume checkpoint does not exist: {resume_path}")
        if not output_dir.is_dir():
            raise FileNotFoundError(f"resume output directory does not exist: {output_dir}")
    else:
        if output_dir.exists():
            raise FileExistsError(f"output directory already exists: {output_dir}")
        if init_path and not pathlib.Path(init_path).is_file():
            raise FileNotFoundError(f"initial checkpoint does not exist: {init_path}")

    data_dir = pathlib.Path(arguments["data_dir"]).resolve()
    data_manifest = load_data_manifest(data_dir)
    sequence_length = int(arguments["sequence_length"])
    if data_manifest["packing"]["sequence_length"] != sequence_length:
        raise ValueError("data and training sequence lengths differ")
    train_dataset = TokenSequenceDataset(data_dir / data_manifest["files"]["train"]["path"], sequence_length)
    validation_dataset = TokenSequenceDataset(
        data_dir / data_manifest["files"]["validation"]["path"], sequence_length
    )

    profile = arguments["profile"]
    layer_indices = () if profile == "d0" else tuple(arguments["layer_indices"])
    model = load_qwen3_model(
        arguments["model_path"],
        dtype=runtime.dtype,
        layer_indices=layer_indices,
        profile="bo" if profile == "d0" else profile,
        receiver_count=arguments["receiver_count"],
        state_size=arguments["state_size"],
        implementation=arguments["implementation"],
        scan_implementation=arguments["scan_implementation"],
        balance_coefficient=arguments["balance_coefficient"],
        attention_implementation=arguments["attention_implementation"],
    )
    model.to(runtime.device)
    model.train()

    extension_ids = model.extension_parameter_ids()
    backbone_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) not in extension_ids
    ]
    extension_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) in extension_ids
    ]
    groups = [
        {"params": backbone_parameters, "lr": arguments["backbone_lr"], "name": "backbone"}
    ]
    if extension_parameters:
        groups.append({"params": extension_parameters, "lr": arguments["extension_lr"], "name": "extension"})
    optimizer = torch.optim.AdamW(
        groups,
        betas=(arguments["beta1"], arguments["beta2"]),
        weight_decay=arguments["weight_decay"],
    )
    base_lrs = [group["lr"] for group in optimizer.param_groups]

    step = tokens = consumed_sequences = 0
    checkpoint: dict[str, Any] | None = None
    if resume_path or init_path:
        checkpoint = torch.load(resume_path or init_path, map_location="cpu", weights_only=False)
        if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError("unsupported checkpoint schema")
        model.load_state_dict(checkpoint["model"], strict=True)
        if resume_path:
            optimizer.load_state_dict(checkpoint["optimizer"])
            _move_optimizer_state(optimizer, runtime.device)
            progress = checkpoint["progress"]
            step = int(progress["step"])
            tokens = int(progress["tokens"])
            consumed_sequences = int(progress["consumed_sequences"])
            _restore_rng(runtime, checkpoint["rng"])

    if not resume_path:
        output_dir.mkdir(parents=True, exist_ok=False)
    log_path = output_dir / "metrics.jsonl"
    effective_tokens = arguments["micro_batch_size"] * arguments["gradient_accumulation"] * sequence_length
    planned_steps = arguments.get("max_steps") or math.ceil(arguments["max_tokens"] / effective_tokens)
    manifest = base_manifest(runtime=runtime, arguments=arguments)
    manifest.update(
        {
            "model": model_identity(arguments["model_path"]),
            "data": {
                "manifest_sha256": sha256_file(data_dir / "manifest.json"),
                "manifest": data_manifest,
            },
            "training": {
                "planned_steps": planned_steps,
                "effective_tokens_per_step": effective_tokens,
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "extension_parameter_count": sum(parameter.numel() for parameter in extension_parameters),
                "resume_contract": "same-stack exact trajectory when deterministic operators permit",
            },
        }
    )
    atomic_write_json(output_dir / "manifest.json", manifest)

    def log(event: dict[str, Any]) -> None:
        encoded = json.dumps(event, sort_keys=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
        print(encoded, flush=True)

    if arguments["run_initial_validation"] and step == 0:
        validation = evaluate(
            model=model,
            runtime=runtime,
            dataset=validation_dataset,
            batch_size=arguments["evaluation_batch_size"],
            max_tokens=arguments["validation_tokens"],
        )
        log({"event": "validation", "phase": "initial", "step": step, **validation})
        model.train()

    batches = deterministic_batches(
        train_dataset,
        batch_size=arguments["micro_batch_size"],
        seed=arguments["seed"],
        consumed_sequences=consumed_sequences,
    )
    optimizer.zero_grad(set_to_none=True)
    while step < planned_steps and tokens < arguments["max_tokens"]:
        step_started = time.monotonic()
        loss_sum = lm_loss_sum = balance_sum = 0.0
        last_metrics: dict[int, dict[str, Any]] = {}
        for _ in range(arguments["gradient_accumulation"]):
            input_ids, consumed_sequences = next(batches)
            input_ids = input_ids.to(runtime.device, non_blocking=True)
            attention_mask = torch.ones_like(input_ids)
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            if output.loss is None or not bool(torch.isfinite(output.loss.detach()).cpu()):
                raise FloatingPointError(f"non-finite loss at step {step + 1}")
            (output.loss / arguments["gradient_accumulation"]).backward()
            loss_sum += float(output.loss.detach().float().cpu())
            lm_loss_sum += float(output.lm_loss.detach().float().cpu())
            balance_sum += float(output.balance_loss.detach().float().cpu())
            last_metrics = output.metrics
            tokens += input_ids.numel()
            del output

        next_step = step + 1
        scale = _learning_rate_scale(
            next_step,
            planned_steps,
            arguments["warmup_ratio"],
            arguments["minimum_lr_ratio"],
        )
        for group, base_lr in zip(optimizer.param_groups, base_lrs, strict=True):
            group["lr"] = base_lr * scale
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), arguments["gradient_clip"])
        coverage = _gradient_coverage(model)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        runtime.synchronize()
        step = next_step
        elapsed = time.monotonic() - step_started
        if step % arguments["log_every"] == 0:
            log(
                {
                    "event": "train",
                    "step": step,
                    "tokens": tokens,
                    "loss": loss_sum / arguments["gradient_accumulation"],
                    "lm_loss": lm_loss_sum / arguments["gradient_accumulation"],
                    "balance_loss": balance_sum / arguments["gradient_accumulation"],
                    "gradient_norm": float(gradient_norm.detach().float().cpu()),
                    "gradient_coverage": coverage,
                    "learning_rates": [group["lr"] for group in optimizer.param_groups],
                    "seconds": elapsed,
                    "tokens_per_second": effective_tokens / elapsed,
                    "peak_memory_bytes": runtime.memory_allocated(),
                    "receiver_metrics": _jsonable_metrics(last_metrics),
                }
            )
        if arguments["checkpoint_every"] and step % arguments["checkpoint_every"] == 0:
            checkpoint_path = _save_checkpoint(
                model=model,
                optimizer=optimizer,
                runtime=runtime,
                output_dir=output_dir,
                step=step,
                tokens=tokens,
                consumed_sequences=consumed_sequences,
                configuration=arguments,
            )
            log({"event": "checkpoint", "step": step, "path": str(checkpoint_path)})

    final_validation = evaluate(
        model=model,
        runtime=runtime,
        dataset=validation_dataset,
        batch_size=arguments["evaluation_batch_size"],
        max_tokens=arguments["validation_tokens"],
    )
    log({"event": "validation", "phase": "final", "step": step, **final_validation})
    final_checkpoint = _save_checkpoint(
        model=model,
        optimizer=optimizer,
        runtime=runtime,
        output_dir=output_dir,
        step=step,
        tokens=tokens,
        consumed_sequences=consumed_sequences,
        configuration=arguments,
    )
    result = {
        "status": "complete",
        "step": step,
        "tokens": tokens,
        "validation": final_validation,
        "checkpoint": str(final_checkpoint),
    }
    atomic_write_json(output_dir / "result.json", result)
    log({"event": "complete", **result})
    return result


def run_probe(runtime: Any, arguments: dict[str, Any]) -> dict[str, Any]:
    """Run the real TIDE operator surface and a two-step Qwen3 backward smoke."""

    torch = runtime.torch
    from .manifest import atomic_write_json
    from .qwen3 import load_qwen3_model
    from .receiver import TideReceiverGroup

    output_dir = pathlib.Path(arguments["output_dir"]).resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")

    hidden_size = 32
    intermediate_size = 64
    group_arguments = {
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "state_size": 8,
        "receiver_count": 4,
        "profile": arguments["profile"],
        "scan_implementation": "reference",
        "ffn_dtype": runtime.dtype,
    }
    reference = TideReceiverGroup(
        **group_arguments,
        implementation="dense-masked-reference",
    )
    packed = TideReceiverGroup(**group_arguments, implementation="packed")
    with torch.no_grad():
        for ffn in reference.ffns:
            ffn.down_proj.weight.normal_(mean=0.0, std=0.02)
    packed.load_state_dict(reference.state_dict(), strict=True)
    reference.to(runtime.device)
    packed.to(runtime.device)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(arguments["seed"])
    fixture = torch.randn(
        arguments["batch_size"],
        arguments["sequence_length"],
        hidden_size,
        generator=generator,
        dtype=torch.float32,
    ).to(device=runtime.device, dtype=runtime.dtype)
    reference_output = reference(fixture, return_artifacts=True)
    packed_output = packed(fixture, return_artifacts=True)
    forward_error = float(
        (reference_output.hidden.float() - packed_output.hidden.float()).abs().max().detach().cpu()
    )
    state_error = float(
        (reference_output.states.float() - packed_output.states.float()).abs().max().detach().cpu()
    )
    tolerance = 2e-5 if runtime.dtype == torch.float32 else 2e-2
    if forward_error > tolerance or state_error > 2e-5:
        raise AssertionError(
            f"packed/reference mismatch: forward={forward_error}, state={state_error}, tolerance={tolerance}"
        )
    reference_output.hidden.float().square().mean().backward()
    packed_output.hidden.float().square().mean().backward()
    active_receiver = int(reference_output.routes.reshape(-1)[0].detach().cpu())
    gradient_error = float(
        (
            reference.ffns[active_receiver].down_proj.weight.grad.float()
            - packed.ffns[active_receiver].down_proj.weight.grad.float()
        )
        .abs()
        .max()
        .detach()
        .cpu()
    )
    if gradient_error > tolerance:
        raise AssertionError(f"packed/reference gradient mismatch: {gradient_error}")

    vectorized = TideReceiverGroup(
        **{**group_arguments, "scan_implementation": "vectorized"},
        implementation="packed",
    )
    vectorized.load_state_dict(packed.state_dict(), strict=True)
    vectorized.to(runtime.device)
    vectorized_output = vectorized(fixture)
    scan_error = float(
        (vectorized_output.hidden.float() - packed_output.hidden.float()).abs().max().detach().cpu()
    )
    if scan_error > tolerance:
        raise AssertionError(f"vectorized/reference scan mismatch: {scan_error}")
    vectorized_output.hidden.float().square().mean().backward()
    runtime.synchronize()

    del reference, packed, vectorized, reference_output, packed_output, vectorized_output, fixture
    model = load_qwen3_model(
        arguments["model_path"],
        dtype=runtime.dtype,
        layer_indices=(arguments["layer_index"],),
        profile=arguments["profile"],
        receiver_count=4,
        state_size=128,
        implementation="packed",
        scan_implementation="vectorized",
        balance_coefficient=0.01,
        attention_implementation=arguments["attention_implementation"],
    )
    extension_ids = model.extension_parameter_ids()
    for parameter in model.base_model.parameters():
        parameter.requires_grad_(id(parameter) in extension_ids)
    extension_parameters = [
        parameter for parameter in model.parameters() if id(parameter) in extension_ids
    ]
    model.to(runtime.device)
    model.train()
    optimizer = torch.optim.AdamW(extension_parameters, lr=1e-4)
    qwen_losses: list[float] = []
    inner_gradient_after_second_step = 0.0
    for probe_step in range(2):
        input_ids = torch.randint(
            low=0,
            high=model.config.vocab_size,
            size=(arguments["batch_size"], arguments["sequence_length"]),
            generator=generator,
            dtype=torch.long,
        ).to(runtime.device)
        output = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), labels=input_ids)
        if output.loss is None or not bool(torch.isfinite(output.loss.detach()).cpu()):
            raise FloatingPointError("Qwen probe produced a non-finite loss")
        output.loss.backward()
        qwen_losses.append(float(output.loss.detach().float().cpu()))
        if probe_step == 1:
            group = model.wrapped_layers[arguments["layer_index"]].receiver_group
            gradients = [
                parameter.grad.detach().float().norm()
                for name, parameter in group.named_parameters()
                if parameter.grad is not None and "down_proj" not in name
            ]
            inner_gradient_after_second_step = float(torch.stack(gradients).sum().cpu()) if gradients else 0.0
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    runtime.synchronize()
    if inner_gradient_after_second_step <= 0.0:
        raise AssertionError("receiver inner parameters did not obtain gradient after leaving zero init")

    report = base_manifest(runtime=runtime, arguments=arguments)
    report.update(
        {
            "status": "passed",
            "checks": {
                "packed_reference_forward_max_abs": forward_error,
                "packed_reference_state_max_abs": state_error,
                "packed_reference_gradient_max_abs": gradient_error,
                "vectorized_scan_forward_max_abs": scan_error,
                "qwen_losses": qwen_losses,
                "inner_gradient_after_second_step": inner_gradient_after_second_step,
                "peak_memory_bytes": runtime.memory_allocated(),
            },
            "model": model_identity(arguments["model_path"]),
        }
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    atomic_write_json(output_dir / "probe.json", report)
    print(json.dumps(report["checks"], indent=2, sort_keys=True), flush=True)
    return report
