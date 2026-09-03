"""Two-process CPU/accelerator parity worker for SettleGraph FP32.

Run ``prepare`` in a CPU-only process, then run ``compare`` in a fresh process
whose requested accelerator has already been isolated by the site launcher.
The processes exchange only CPU Tensor artifacts and a CPU checkpoint.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping as TypingMapping, Optional

import torch
from torch import Tensor

from tide.builders import build_single_layer
from tide.checkpoint import load_checkpoint, save_checkpoint
from tide.engine import SettleGraph
from tide.equivalence import (
    CPU_NPU_FLOAT32_TOLERANCE,
    SAME_BACKEND_FLOAT32_TOLERANCE,
    compare_nested,
    validate_trace_invariants,
)
from tide.ops import safe_module_key
from tide.plan import bind_dtypes
from tide.runtime import RuntimeRequest, resolve_runtime


SCHEMA = "tide.settlegraph.backend-parity.v1"


def _plan():
    plan = build_single_layer(receiver_count=2, k=1, d_model=2)
    nodes = tuple(
        dataclasses.replace(
            node,
            state_shape=(2,),
            state_owner=node.node_id,
            update={
                "type": "ema",
                "formula_id": "state.ema.v1",
                "state_dim": 2,
                "decay": 0.5,
                "state_shape": [2],
            },
            selector_read_shape=(1,),
            selector_read={
                "type": "content_state_linear",
                "formula_id": "TEST-READ-PROJ-V1",
                "out_dim": 1,
                "output_shape": [1],
            },
            ffn_read={
                "type": "state_default",
                "formula_id": "read.ffn.ema.v1",
                "output_shape": [2],
            },
            node_compute={
                "type": "affine_residual",
                "formula_id": "TEST-NODE-AFFINE-V1",
                "bias": True,
                "output_shape": [2],
            },
            emit={
                "type": "hst",
                "formula_id": "emit.hst.v1",
                "zeta": 0.7,
                "output_shape": [2],
            },
        )
        for node in plan.nodes
    )
    region = dataclasses.replace(
        plan.regions[0],
        profile="BO",
        selector_timing="post",
        score={
            "type": "linear",
            "formula_id": "TEST-SCORE-LINEAR-V1",
            "bias": True,
        },
    )
    return dataclasses.replace(
        plan, nodes=nodes, regions=(region,)
    ).validate()


def _typed_plan():
    return bind_dtypes(
        _plan(),
        hidden="float32",
        parameter="float32",
        state="float32",
        readout="float32",
    )


def _configure(model: SettleGraph) -> None:
    selector = model.selector("region.0")
    with torch.no_grad():
        for index, node_id in enumerate(model.plan.regions[0].node_ids):
            linear = selector.linears[safe_module_key(node_id)]
            linear.weight.zero_()
            linear.bias.fill_(2.0 if index == 0 else -2.0)


def _model(
    runtime: Any,
    state_dict: Optional[TypingMapping[str, Tensor]] = None,
):
    model = SettleGraph(_typed_plan()).to(
        device=runtime.device, dtype=runtime.dtype
    )
    _configure(model)
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
    return model


def _source_fixture() -> Dict[str, Tensor]:
    return {
        "hidden": torch.tensor(
            [
                [[0.25, -0.5], [1.0, 0.75], [-0.25, 1.5]],
                [[-0.75, 0.5], [0.125, -1.0], [9.0, 8.0]],
            ],
            dtype=torch.float32,
        ),
        "execution": torch.tensor(
            [[True, True, True], [True, True, False]], dtype=torch.bool
        ),
        "positions": torch.tensor(
            [[0, 1, 2], [0, 1, 99]], dtype=torch.int64
        ),
        "cotangent": torch.tensor(
            [
                [[0.5, -0.25], [0.75, 0.125], [-0.5, 0.25]],
                [[0.25, 0.5], [-0.75, 1.0], [0.0, 0.0]],
            ],
            dtype=torch.float32,
        ),
    }


def _to_device(source: Mapping[str, Tensor], runtime: Any) -> Dict[str, Tensor]:
    return {
        name: value.to(device=runtime.device)
        for name, value in source.items()
    }


def _run(model: SettleGraph, source: Mapping[str, Tensor], runtime: Any):
    values = _to_device(source, runtime)
    hidden = values["hidden"].detach().clone().requires_grad_(True)
    token_result = model.prefill(
        hidden,
        values["execution"],
        ["long", "short"],
        values["positions"],
        detach_at_end=False,
        record_trace=True,
    )
    token_loss = (token_result.output * values["cotangent"]).sum()
    token_loss = token_loss + 0.2 * token_result.balance_loss
    token_loss.backward()
    token_input_gradient = hidden.grad.detach().clone()
    token_parameter_gradients = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }

    model.zero_grad(set_to_none=True)
    region_hidden = values["hidden"].detach().clone().requires_grad_(True)
    region_result = model.prefill_region_major(
        region_hidden,
        values["execution"],
        ["long", "short"],
        values["positions"],
        detach_at_end=False,
        record_trace=True,
    )
    region_loss = (region_result.output * values["cotangent"]).sum()
    region_loss = region_loss + 0.2 * region_result.balance_loss
    region_loss.backward()
    region_parameter_gradients = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }
    runtime.synchronize()

    assert token_result.trace is not None and region_result.trace is not None
    executed = (
        ("long", 0),
        ("long", 1),
        ("long", 2),
        ("short", 0),
        ("short", 1),
    )
    validate_trace_invariants(
        model.plan,
        token_result.trace,
        executed,
        tolerance=SAME_BACKEND_FLOAT32_TOLERANCE,
    )
    validate_trace_invariants(
        model.plan,
        region_result.trace,
        executed,
        tolerance=SAME_BACKEND_FLOAT32_TOLERANCE,
    )
    compare_nested(
        token_result,
        region_result,
        tolerance=SAME_BACKEND_FLOAT32_TOLERANCE,
    ).require_pass()
    compare_nested(
        token_input_gradient,
        region_hidden.grad,
        tolerance=SAME_BACKEND_FLOAT32_TOLERANCE,
    ).require_pass()
    compare_nested(
        token_parameter_gradients,
        region_parameter_gradients,
        tolerance=SAME_BACKEND_FLOAT32_TOLERANCE,
    ).require_pass()
    _require_finite_on_backend(region_result)
    _require_finite_on_backend(region_hidden.grad)
    _require_finite_on_backend(region_parameter_gradients)
    return {
        "result": _plain(region_result),
        "input_gradient": region_hidden.grad.detach().cpu(),
        "parameter_gradients": _plain(region_parameter_gradients),
    }


def _checkpoint_path(artifact: Path) -> Path:
    return artifact.with_name(artifact.name + ".checkpoint.pt")


def _prepare_checkpoint(
    artifact: Path,
    model_state: Mapping[str, Tensor],
    source: Mapping[str, Tensor],
    runtime: Any,
) -> Dict[str, Any]:
    model = _model(runtime, model_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    values = _to_device(source, runtime)
    first = model.prefill_region_major(
        values["hidden"][:1, :1],
        values["execution"][:1, :1],
        ["checkpoint-sequence"],
        values["positions"][:1, :1],
        detach_at_end=False,
    )
    optimizer.zero_grad(set_to_none=True)
    first.output.square().sum().backward()
    optimizer.step()
    checkpoint = _checkpoint_path(artifact)
    saved = save_checkpoint(
        checkpoint,
        model=model,
        typed_plan=_typed_plan(),
        state=first.state,
        optimizer=optimizer,
        progress={"global_step": 1, "token_count": 1},
        training_state={"statistics_window_boundary": True},
    )
    continuation = model.prefill_region_major(
        values["hidden"][:1, 1:2],
        values["execution"][:1, 1:2],
        ["checkpoint-sequence"],
        values["positions"][:1, 1:2],
        state=first.state,
        detach_at_end=False,
    )
    return {
        "filename": checkpoint.name,
        "sha256": saved.sha256,
        "continuation": _plain(continuation),
    }


def _compare_checkpoint(
    artifact: Path,
    record: Mapping[str, Any],
    source: Mapping[str, Tensor],
    runtime: Any,
) -> None:
    model = _model(runtime)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    checkpoint = artifact.with_name(str(record["filename"]))
    loaded = load_checkpoint(
        checkpoint,
        model=model,
        typed_plan=_typed_plan(),
        mode="resume",
        optimizer=optimizer,
        expected_sha256=str(record["sha256"]),
    )
    values = _to_device(source, runtime)
    continuation = model.prefill_region_major(
        values["hidden"][:1, 1:2],
        values["execution"][:1, 1:2],
        ["checkpoint-sequence"],
        values["positions"][:1, 1:2],
        state=loaded.state,
        detach_at_end=False,
    )
    runtime.synchronize()
    _require_finite_on_backend(continuation)
    compare_nested(
        record["continuation"],
        _plain(continuation),
        tolerance=CPU_NPU_FLOAT32_TOLERANCE,
    ).require_pass()


def _plain(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().contiguous()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "__type__": type(value).__name__,
            **{
                field.name: _plain(getattr(value, field.name))
                for field in dataclasses.fields(value)
            },
        }
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_plain(item) for item in value)
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _require_finite_on_backend(value: Any) -> None:
    if isinstance(value, Tensor):
        if value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
            raise AssertionError("backend result contains NaN or infinity")
        return
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            _require_finite_on_backend(getattr(value, field.name))
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _require_finite_on_backend(item)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _require_finite_on_backend(item)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cpu_state_dict(model: SettleGraph) -> Dict[str, Tensor]:
    return {
        name: value.detach().cpu().contiguous()
        for name, value in model.state_dict().items()
    }


def prepare(artifact: Path) -> Dict[str, Any]:
    runtime = resolve_runtime(RuntimeRequest("cpu", 0, "float32", 20260903))
    typed = _typed_plan()
    model = _model(runtime)
    source = _source_fixture()
    model_state = _cpu_state_dict(model)
    expected = _run(model, source, runtime)
    checkpoint = _prepare_checkpoint(
        artifact, model_state, source, runtime
    )
    bundle = {
        "schema": SCHEMA,
        "logical_plan_hash": typed.logical_hash(),
        "typed_plan_hash": typed.typed_hash(),
        "source": source,
        "model_state": model_state,
        "expected": expected,
        "checkpoint": checkpoint,
    }
    artifact.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, artifact)
    return {
        "mode": "prepare",
        "artifact_sha256": _sha256(artifact),
        "checkpoint_sha256": checkpoint["sha256"],
        "logical_plan_hash": typed.logical_hash(),
        "typed_plan_hash": typed.typed_hash(),
        "runtime": runtime.to_dict(),
    }


def compare(artifact: Path, backend: str, device_index: int) -> Dict[str, Any]:
    runtime = resolve_runtime(
        RuntimeRequest(backend, device_index, "float32", 20260903)
    )
    bundle = torch.load(artifact, map_location="cpu", weights_only=True)
    if not isinstance(bundle, Mapping) or bundle.get("schema") != SCHEMA:
        raise ValueError("parity artifact schema is incompatible")
    typed = _typed_plan()
    if bundle.get("logical_plan_hash") != typed.logical_hash():
        raise ValueError("parity artifact logical Plan hash mismatch")
    if bundle.get("typed_plan_hash") != typed.typed_hash():
        raise ValueError("parity artifact typed Plan hash mismatch")
    model = _model(runtime, bundle["model_state"])
    actual = _run(model, bundle["source"], runtime)
    report = compare_nested(
        bundle["expected"],
        actual,
        tolerance=CPU_NPU_FLOAT32_TOLERANCE,
    )
    report.require_pass()
    _compare_checkpoint(
        artifact, bundle["checkpoint"], bundle["source"], runtime
    )
    maximum_absolute = max(
        (item.max_absolute_error for item in report.tensors), default=0.0
    )
    maximum_relative = max(
        (item.max_relative_error for item in report.tensors), default=0.0
    )
    return {
        "mode": "compare",
        "artifact_sha256": _sha256(artifact),
        "logical_plan_hash": typed.logical_hash(),
        "typed_plan_hash": typed.typed_hash(),
        "max_absolute_error": maximum_absolute,
        "max_relative_error": maximum_relative,
        "runtime": runtime.to_dict(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="prepare or compare a CPU/accelerator SettleGraph fixture"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--artifact", type=Path, required=True)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--artifact", type=Path, required=True)
    compare_parser.add_argument(
        "--backend", choices=("cuda", "npu"), required=True
    )
    compare_parser.add_argument("--device-index", type=int, default=0)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        report = prepare(args.artifact)
    else:
        report = compare(args.artifact, args.backend, args.device_index)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
