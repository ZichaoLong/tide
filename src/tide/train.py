"""Training, milestone diagnostics, and live-device probes for TIDE experiments."""

from __future__ import annotations

import json
import math
import os
import pathlib
import random
import re
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from .data import TokenSequenceDataset, deterministic_batches, load_data_manifest
from .manifest import atomic_write_json, base_manifest, model_identity, sha256_file


CHECKPOINT_SCHEMA = "tide-training-v1"
CHECKPOINT_NAME_PATTERN = re.compile(
    r"(?:step-\d+|token-\d+-step-\d+)\.pt\Z"
)


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


def _scheduled_learning_rate_scale(
    *,
    schedule: str,
    step: int,
    tokens: int,
    total_steps: int,
    warmup_ratio: float,
    warmup_tokens: int,
    minimum_ratio: float,
) -> float:
    if schedule == "warmup-cosine":
        return _learning_rate_scale(step, total_steps, warmup_ratio, minimum_ratio)
    if schedule == "warmup-stable":
        return 1.0 if warmup_tokens == 0 else min(1.0, tokens / warmup_tokens)
    raise ValueError(f"unknown learning-rate schedule: {schedule}")


def _parameter_gradient_summary(parameters: Sequence[Any]) -> dict[str, Any]:
    import torch

    gradients = [parameter.grad.detach() for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return {"norm": None, "parameter_tensors": len(parameters), "tensors_with_grad": 0}
    norms = torch.stack([gradient.norm().float() for gradient in gradients])
    return {
        "norm": float(norms.norm().cpu()),
        "parameter_tensors": len(parameters),
        "tensors_with_grad": len(gradients),
    }


def _gradient_coverage(model: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for layer_index, wrapped in model.wrapped_layers.items():
        group = wrapped.receiver_group
        result[str(layer_index)] = {
            "router": _parameter_gradient_summary(list(group.router.parameters())),
            "observe": _parameter_gradient_summary(
                [parameter for observer in group.observers for parameter in observer.parameters()]
            ),
            "state": _parameter_gradient_summary(
                [group.decay_logits]
                + [
                    parameter
                    for projection in group.state_projections
                    for parameter in projection.parameters()
                ]
            ),
            "ffn_output": _parameter_gradient_summary(
                [receiver.down_proj.weight for receiver in group.ffns]
            ),
        }
    for layer_index, moe in model.moe_layers.items():
        result[str(layer_index)] = {
            "router": _parameter_gradient_summary(list(moe.router.parameters())),
            "expert_output": _parameter_gradient_summary(
                [expert.down_proj.weight for expert in moe.experts]
            ),
        }
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


def _entropy(probabilities: Any) -> float:
    positive = probabilities[probabilities > 0]
    if positive.numel() == 0:
        return 0.0
    return float((-(positive * positive.log()).sum()).cpu())


def _jensen_shannon(left: Any, right: Any) -> float:
    middle = 0.5 * (left + right)

    def divergence(value: Any) -> Any:
        positive = value > 0
        return (value[positive] * (value[positive] / middle[positive]).log()).sum()

    return float((0.5 * divergence(left) + 0.5 * divergence(right)).cpu())


def _diagnostic_scalar_metrics(report: Mapping[str, Any]) -> dict[str, float]:
    metrics = {
        "diagnostic/normal_loss": float(report["normal_loss"]),
    }
    path = report.get("path")
    if isinstance(path, Mapping):
        metrics.update(
            {
                "diagnostic/path_entropy": float(path["entropy"]),
                "diagnostic/effective_paths": float(path["effective_count"]),
                "diagnostic/path_churn_from_initial": float(path["churn_from_initial"]),
            }
        )
    for name, values in report.get("state_interventions", {}).items():
        metrics[f"diagnostic/{name}_delta_loss"] = float(values["delta_loss"])
    return metrics


def run_fixed_diagnostics(
    *,
    model: Any,
    runtime: Any,
    dataset: TokenSequenceDataset,
    batch_size: int,
    max_tokens: int,
    output_dir: pathlib.Path,
    label: str,
) -> dict[str, Any]:
    """Measure fixed-prefix routes and causal state interventions."""

    torch = runtime.torch
    maximum_sequences = min(len(dataset), max_tokens // dataset.sequence_length)
    if maximum_sequences < 1:
        raise ValueError("diagnostic token budget must contain at least one complete sequence")
    layer_indices = sorted([*model.wrapped_layers, *model.moe_layers])
    route_chunks: dict[int, list[Any]] = {index: [] for index in layer_indices}
    probability_sums: dict[int, Any] = {}
    total_loss = 0.0
    total_sequences = 0
    intervention_loss = {name: 0.0 for name in ("no_read", "clear", "shuffle")}
    intervention_layers = {
        index: (dataset.sequence_length // 2,) for index in model.wrapped_layers
    }
    started = time.monotonic()
    model.eval()
    with torch.no_grad():
        for start in range(0, maximum_sequences, batch_size):
            indices = range(start, min(start + batch_size, maximum_sequences))
            input_ids = torch.stack([dataset[index] for index in indices]).to(runtime.device)
            attention_mask = torch.ones_like(input_ids)
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
                tide_return_artifacts=True,
            )
            batch_sequences = input_ids.shape[0]
            total_loss += float(output.lm_loss.float().cpu()) * batch_sequences
            total_sequences += batch_sequences
            for layer_index in layer_indices:
                artifact = output.artifacts[layer_index]
                routes = artifact.routes
                probabilities = artifact.probabilities
                if routes is None or probabilities is None:
                    raise RuntimeError(f"layer {layer_index} omitted requested route artifacts")
                route_chunks[layer_index].append(routes.to(device="cpu", dtype=torch.int16))
                probability_sum = probabilities.float().sum(dim=(0, 1)).cpu()
                probability_sums[layer_index] = (
                    probability_sum
                    if layer_index not in probability_sums
                    else probability_sums[layer_index] + probability_sum
                )

            if intervention_layers:
                variants = {
                    "no_read": {"tide_read_state": False},
                    "clear": {"tide_clear_positions": intervention_layers},
                    "shuffle": {"tide_shuffle_positions": intervention_layers},
                }
                for name, extra in variants.items():
                    changed = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids,
                        **extra,
                    )
                    intervention_loss[name] += (
                        float(changed.lm_loss.float().cpu()) * batch_sequences
                    )
    runtime.synchronize()

    normal_loss = total_loss / total_sequences
    routes = {index: torch.cat(chunks, dim=0) for index, chunks in route_chunks.items()}
    token_count = total_sequences * dataset.sequence_length
    mean_probabilities = {
        index: probability_sums[index] / token_count for index in layer_indices
    }
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = diagnostics_dir / f"probe-{label}.pt"
    _atomic_torch_save(
        torch,
        {
            "schema": "tide-fixed-probe-v1",
            "label": label,
            "tokens": token_count,
            "routes": routes,
            "mean_probabilities": mean_probabilities,
        },
        artifact_path,
    )

    layer_reports: dict[str, Any] = {}
    for layer_index in layer_indices:
        receiver_count = int(mean_probabilities[layer_index].numel())
        route_distribution = torch.bincount(
            routes[layer_index].reshape(-1).long(), minlength=receiver_count
        ).float()
        route_distribution /= route_distribution.sum()
        entropy = _entropy(route_distribution)
        layer_reports[str(layer_index)] = {
            "route_distribution": route_distribution.tolist(),
            "mean_probability": mean_probabilities[layer_index].tolist(),
            "route_entropy": entropy,
            "effective_receivers": math.exp(entropy),
        }

    report: dict[str, Any] = {
        "label": label,
        "tokens": token_count,
        "normal_loss": normal_loss,
        "seconds": time.monotonic() - started,
        "artifact": str(artifact_path),
        "layers": layer_reports,
        "state_interventions": {
            name: {
                "loss": value / total_sequences,
                "delta_loss": value / total_sequences - normal_loss,
            }
            for name, value in intervention_loss.items()
        }
        if intervention_layers
        else {},
    }
    if layer_indices:
        current_path = torch.stack(
            [routes[index].reshape(-1).long() for index in layer_indices], dim=1
        )
        _, path_counts = torch.unique(current_path, dim=0, return_counts=True)
        path_distribution = path_counts.float() / path_counts.sum()
        path_entropy = _entropy(path_distribution)
        initial_path = diagnostics_dir / "probe-initial.pt"
        if label == "initial":
            churn = 0.0
            layer_churn = {str(index): 0.0 for index in layer_indices}
            distribution_drift = {str(index): 0.0 for index in layer_indices}
        else:
            if not initial_path.is_file():
                raise FileNotFoundError("initial fixed-probe routes are missing")
            initial = torch.load(initial_path, map_location="cpu", weights_only=False)
            initial_routes = initial["routes"]
            initial_complete_path = torch.stack(
                [initial_routes[index].reshape(-1).long() for index in layer_indices], dim=1
            )
            if initial_complete_path.shape != current_path.shape:
                raise ValueError("fixed-probe route shape changed")
            churn = float((current_path != initial_complete_path).any(dim=1).float().mean())
            layer_churn = {
                str(index): float(
                    (routes[index] != initial_routes[index]).float().mean()
                )
                for index in layer_indices
            }
            distribution_drift = {}
            for index in layer_indices:
                receiver_count = int(mean_probabilities[index].numel())
                current_distribution = torch.bincount(
                    routes[index].reshape(-1).long(), minlength=receiver_count
                ).float()
                current_distribution /= current_distribution.sum()
                original_distribution = torch.bincount(
                    initial_routes[index].reshape(-1).long(), minlength=receiver_count
                ).float()
                original_distribution /= original_distribution.sum()
                distribution_drift[str(index)] = _jensen_shannon(
                    original_distribution, current_distribution
                )
        report["path"] = {
            "layer_order": layer_indices,
            "unique_count": int(path_counts.numel()),
            "entropy": path_entropy,
            "effective_count": math.exp(path_entropy),
            "churn_from_initial": churn,
            "layer_churn_from_initial": layer_churn,
            "route_distribution_js_from_initial": distribution_drift,
        }
    return report


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
    milestone_tokens: int | None = None,
) -> pathlib.Path:
    torch = runtime.torch
    if milestone_tokens is None:
        name = f"step-{step:06d}.pt"
    else:
        name = f"token-{milestone_tokens:010d}-step-{step:06d}.pt"
    path = output_dir / "checkpoints" / name
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
        {
            "schema": CHECKPOINT_SCHEMA,
            "path": path.name,
            "step": step,
            "tokens": tokens,
            "milestone_tokens": milestone_tokens,
            "retention": {
                "keep_last": configuration["checkpoint_keep_last"],
            },
        },
    )
    _prune_superseded_checkpoints(
        output_dir=output_dir,
        latest_checkpoint=path,
        keep_last=configuration["checkpoint_keep_last"],
    )
    return path


def _prune_superseded_checkpoints(
    *,
    output_dir: pathlib.Path,
    latest_checkpoint: pathlib.Path,
    keep_last: int,
) -> None:
    """Retain full resume state for the newest checkpoints in one run."""

    if keep_last < 0:
        raise ValueError("checkpoint-keep-last must be nonnegative")
    if keep_last == 0:
        return

    checkpoint_dir = output_dir / "checkpoints"
    latest_checkpoint = latest_checkpoint.resolve()
    candidates = [
        candidate
        for candidate in checkpoint_dir.iterdir()
        if candidate.is_file()
        and not candidate.is_symlink()
        and CHECKPOINT_NAME_PATTERN.fullmatch(candidate.name)
    ]
    if latest_checkpoint not in [candidate.resolve() for candidate in candidates]:
        raise RuntimeError("latest checkpoint is not a regular managed checkpoint")
    ordered = sorted(
        candidates,
        key=lambda candidate: (candidate.stat().st_mtime_ns, candidate.name),
        reverse=True,
    )
    retained = {latest_checkpoint}
    for candidate in ordered:
        resolved = candidate.resolve()
        if resolved != latest_checkpoint and len(retained) < keep_last:
            retained.add(resolved)

    audit_path = checkpoint_dir / "retention.jsonl"
    for candidate in ordered:
        resolved = candidate.resolve()
        if resolved in retained:
            continue
        stat = candidate.stat()
        candidate.unlink()
        event = {
            "schema_version": 1,
            "event": "checkpoint_pruned",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "path": candidate.name,
            "bytes": stat.st_size,
            "reason": "superseded_by_latest",
            "latest": latest_checkpoint.name,
            "keep_last": keep_last,
        }
        with audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def _run_training(runtime: Any, arguments: dict[str, Any]) -> dict[str, Any]:
    torch = runtime.torch
    from .qwen3 import load_qwen3_model
    from .tracking import TrackioProjection

    arguments = {**arguments}
    arguments.setdefault("checkpoint_keep_last", 1)
    if arguments["checkpoint_keep_last"] < 0:
        raise ValueError("checkpoint-keep-last must be nonnegative")
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

    effective_tokens = (
        arguments["micro_batch_size"] * arguments["gradient_accumulation"] * sequence_length
    )
    planned_steps = arguments.get("max_steps") or math.ceil(
        arguments["max_tokens"] / effective_tokens
    )
    planned_tokens = planned_steps * effective_tokens
    milestones = list(arguments["milestone_tokens"])
    if milestones and milestones[-1] > arguments["max_tokens"]:
        raise ValueError("a milestone exceeds max-tokens")
    if arguments["diagnostic_tokens"] and arguments["diagnostic_tokens"] < sequence_length:
        raise ValueError("diagnostic-tokens must contain at least one complete sequence")
    if (
        arguments["initialization"] == "random"
        and len(train_dataset) * sequence_length < planned_tokens
    ):
        raise ValueError(
            "random pretraining data is smaller than the rounded training budget; "
            "prepare enough data to avoid repeating sequences"
        )

    profile = arguments["profile"]
    layer_indices = () if profile == "d0" else tuple(arguments["layer_indices"])
    model = load_qwen3_model(
        arguments["model_path"],
        dtype=runtime.dtype,
        layer_indices=layer_indices,
        profile=profile,
        receiver_count=arguments["receiver_count"],
        expert_count=arguments["expert_count"],
        state_size=arguments["state_size"],
        implementation=arguments["implementation"],
        scan_implementation=arguments["scan_implementation"],
        balance_coefficient=arguments["balance_coefficient"],
        router_z_coefficient=arguments["router_z_coefficient"],
        attention_implementation=arguments["attention_implementation"],
        initialization=arguments["initialization"],
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
    manifest = base_manifest(runtime=runtime, arguments=arguments)
    manifest.update(
        {
            "status": "running",
            "model": {
                **model_identity(
                    arguments["model_path"],
                    include_weights=arguments["initialization"] == "pretrained",
                ),
                "initialization": arguments["initialization"],
                "source_contract": (
                    "checkpoint weights, config, and tokenizer"
                    if arguments["initialization"] == "pretrained"
                    else "config and tokenizer only; checkpoint weights were not loaded"
                ),
            },
            "data": {
                "manifest_sha256": sha256_file(data_dir / "manifest.json"),
                "manifest": data_manifest,
            },
            "training": {
                "planned_steps": planned_steps,
                "rounded_planned_tokens": planned_tokens,
                "effective_tokens_per_step": effective_tokens,
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "extension_parameter_count": sum(parameter.numel() for parameter in extension_parameters),
                "architectural_added_parameter_count": model.added_parameter_count(),
                "resume_contract": "same-stack exact trajectory when deterministic operators permit",
            },
            "tracking": {
                "mode": arguments["tracking"],
                "project": arguments["trackio_project"],
                "status_artifact": "tracking.json",
                "authority": "manifest.json + metrics.jsonl + result.json",
            },
        }
    )
    atomic_write_json(output_dir / "manifest.json", manifest)

    run_name = arguments.get("run_name") or output_dir.name
    tracker = TrackioProjection(
        output_dir,
        mode=arguments["tracking"],
        project=arguments["trackio_project"],
        run_name=run_name,
        config=arguments,
        resume=bool(resume_path),
    )

    def log(
        event: dict[str, Any],
        *,
        trackio_metrics: Mapping[str, int | float] | None = None,
    ) -> None:
        encoded = json.dumps(event, sort_keys=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
        print(encoded, flush=True)
        if trackio_metrics:
            tracker.log(trackio_metrics, step=int(event.get("step", 0)))

    last_validation_step: int | None = None
    last_diagnostic_step: int | None = None
    if arguments["run_initial_validation"] and step == 0:
        validation = evaluate(
            model=model,
            runtime=runtime,
            dataset=validation_dataset,
            batch_size=arguments["evaluation_batch_size"],
            max_tokens=arguments["validation_tokens"],
        )
        log(
            {"event": "validation", "phase": "initial", "step": step, **validation},
            trackio_metrics={
                "validation/loss": validation["loss"],
                "validation/perplexity": validation["perplexity"],
            },
        )
        last_validation_step = step
        model.train()
    if arguments["diagnostic_tokens"] and step == 0:
        diagnostics = run_fixed_diagnostics(
            model=model,
            runtime=runtime,
            dataset=validation_dataset,
            batch_size=arguments["evaluation_batch_size"],
            max_tokens=arguments["diagnostic_tokens"],
            output_dir=output_dir,
            label="initial",
        )
        log(
            {"event": "diagnostic", "phase": "initial", "step": step, **diagnostics},
            trackio_metrics=_diagnostic_scalar_metrics(diagnostics),
        )
        last_diagnostic_step = step
        model.train()

    batches = deterministic_batches(
        train_dataset,
        batch_size=arguments["micro_batch_size"],
        seed=arguments["seed"],
        consumed_sequences=consumed_sequences,
    )
    optimizer.zero_grad(set_to_none=True)
    pending_milestones = [target for target in milestones if target > tokens]
    last_checkpoint: pathlib.Path | None = None
    last_checkpoint_step: int | None = None
    while step < planned_steps and tokens < arguments["max_tokens"]:
        step_started = time.monotonic()
        loss_sum = lm_loss_sum = balance_sum = router_z_sum = 0.0
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
            router_z_sum += float(output.router_z_loss.detach().float().cpu())
            last_metrics = output.metrics
            tokens += input_ids.numel()
            del output

        next_step = step + 1
        scale = _scheduled_learning_rate_scale(
            schedule=arguments["lr_schedule"],
            step=next_step,
            tokens=tokens,
            total_steps=planned_steps,
            warmup_ratio=arguments["warmup_ratio"],
            warmup_tokens=arguments["warmup_tokens"],
            minimum_ratio=arguments["minimum_lr_ratio"],
        )
        for group, base_lr in zip(optimizer.param_groups, base_lrs, strict=True):
            group["lr"] = base_lr * scale
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), arguments["gradient_clip"])
        should_log = next_step == 1 or next_step % arguments["log_every"] == 0
        coverage = _gradient_coverage(model) if should_log else None
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        runtime.synchronize()
        step = next_step
        elapsed = time.monotonic() - step_started
        if should_log:
            train_event = {
                    "event": "train",
                    "step": step,
                    "tokens": tokens,
                    "loss": loss_sum / arguments["gradient_accumulation"],
                    "lm_loss": lm_loss_sum / arguments["gradient_accumulation"],
                    "balance_loss": balance_sum / arguments["gradient_accumulation"],
                    "router_z_loss": router_z_sum / arguments["gradient_accumulation"],
                    "gradient_norm": float(gradient_norm.detach().float().cpu()),
                    "gradient_coverage": coverage,
                    "learning_rates": [group["lr"] for group in optimizer.param_groups],
                    "seconds": elapsed,
                    "tokens_per_second": effective_tokens / elapsed,
                    "peak_memory_bytes": runtime.memory_allocated(),
                    "receiver_metrics": _jsonable_metrics(last_metrics),
                }
            trackio_train = {
                "train/loss": train_event["loss"],
                "train/lm_loss": train_event["lm_loss"],
                "train/balance_loss": train_event["balance_loss"],
                "train/router_z_loss": train_event["router_z_loss"],
                "train/gradient_norm": train_event["gradient_norm"],
                "progress/tokens": tokens,
                "performance/tokens_per_second": train_event["tokens_per_second"],
            }
            for group in optimizer.param_groups:
                trackio_train[f"learning_rate/{group['name']}"] = group["lr"]
            if train_event["peak_memory_bytes"] is not None:
                trackio_train["performance/peak_memory_bytes"] = train_event[
                    "peak_memory_bytes"
                ]
            log(train_event, trackio_metrics=trackio_train)

        milestone_saved = False
        while pending_milestones and tokens >= pending_milestones[0]:
            target = pending_milestones.pop(0)
            checkpoint_path = _save_checkpoint(
                model=model,
                optimizer=optimizer,
                runtime=runtime,
                output_dir=output_dir,
                step=step,
                tokens=tokens,
                consumed_sequences=consumed_sequences,
                configuration=arguments,
                milestone_tokens=target,
            )
            last_checkpoint = checkpoint_path
            last_checkpoint_step = step
            milestone_saved = True
            log(
                {
                    "event": "checkpoint",
                    "phase": "milestone",
                    "step": step,
                    "tokens": tokens,
                    "milestone_tokens": target,
                    "path": str(checkpoint_path),
                }
            )
            validation = evaluate(
                model=model,
                runtime=runtime,
                dataset=validation_dataset,
                batch_size=arguments["evaluation_batch_size"],
                max_tokens=arguments["validation_tokens"],
            )
            log(
                {
                    "event": "validation",
                    "phase": "milestone",
                    "milestone_tokens": target,
                    "step": step,
                    **validation,
                },
                trackio_metrics={
                    "validation/loss": validation["loss"],
                    "validation/perplexity": validation["perplexity"],
                },
            )
            last_validation_step = step
            if arguments["diagnostic_tokens"]:
                diagnostics = run_fixed_diagnostics(
                    model=model,
                    runtime=runtime,
                    dataset=validation_dataset,
                    batch_size=arguments["evaluation_batch_size"],
                    max_tokens=arguments["diagnostic_tokens"],
                    output_dir=output_dir,
                    label=f"token-{target:010d}",
                )
                log(
                    {
                        "event": "diagnostic",
                        "phase": "milestone",
                        "milestone_tokens": target,
                        "step": step,
                        **diagnostics,
                    },
                    trackio_metrics=_diagnostic_scalar_metrics(diagnostics),
                )
                last_diagnostic_step = step
            model.train()

        if (
            not milestone_saved
            and arguments["checkpoint_every"]
            and step % arguments["checkpoint_every"] == 0
        ):
            last_checkpoint = _save_checkpoint(
                model=model,
                optimizer=optimizer,
                runtime=runtime,
                output_dir=output_dir,
                step=step,
                tokens=tokens,
                consumed_sequences=consumed_sequences,
                configuration=arguments,
            )
            last_checkpoint_step = step
            log({"event": "checkpoint", "step": step, "path": str(last_checkpoint)})

    if last_validation_step == step:
        final_validation = validation
    else:
        final_validation = evaluate(
            model=model,
            runtime=runtime,
            dataset=validation_dataset,
            batch_size=arguments["evaluation_batch_size"],
            max_tokens=arguments["validation_tokens"],
        )
        log(
            {"event": "validation", "phase": "final", "step": step, **final_validation},
            trackio_metrics={
                "validation/loss": final_validation["loss"],
                "validation/perplexity": final_validation["perplexity"],
            },
        )
    if arguments["diagnostic_tokens"] and last_diagnostic_step != step:
        diagnostics = run_fixed_diagnostics(
            model=model,
            runtime=runtime,
            dataset=validation_dataset,
            batch_size=arguments["evaluation_batch_size"],
            max_tokens=arguments["diagnostic_tokens"],
            output_dir=output_dir,
            label="final",
        )
        log(
            {"event": "diagnostic", "phase": "final", "step": step, **diagnostics},
            trackio_metrics=_diagnostic_scalar_metrics(diagnostics),
        )
    if last_checkpoint_step == step and last_checkpoint is not None:
        final_checkpoint = last_checkpoint
    else:
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
    tracker.finish()
    return result


def run_training(runtime: Any, arguments: dict[str, Any]) -> dict[str, Any]:
    """Run training while keeping the durable manifest lifecycle truthful."""

    output_dir = pathlib.Path(arguments["output_dir"]).resolve()
    output_preexisted = output_dir.exists()
    try:
        result = _run_training(runtime, arguments)
    except BaseException as error:
        from .tracking import finish_active_projection

        try:
            finish_active_projection()
        except BaseException:
            pass
        if output_dir.is_dir() and (arguments.get("resume") or not output_preexisted):
            status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            failure = {
                "status": status,
                "ended_utc": datetime.now(timezone.utc).isoformat(),
                "error_type": type(error).__name__,
            }
            atomic_write_json(output_dir / "failure.json", failure)
            manifest_path = output_dir / "manifest.json"
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                manifest = {}
            manifest.update(failure)
            atomic_write_json(manifest_path, manifest)
        raise

    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "status": "completed",
            "ended_utc": datetime.now(timezone.utc).isoformat(),
            "result_artifact": "result.json",
        }
    )
    atomic_write_json(manifest_path, manifest)
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
        expert_count=8,
        state_size=128,
        implementation="packed",
        scan_implementation="vectorized",
        balance_coefficient=0.01,
        router_z_coefficient=0.001,
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
