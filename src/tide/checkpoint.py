"""Versioned, portable checkpoints for the eager SettleGraph reference.

Tensor artifacts are stored on CPU.  Loading validates Plan identity and the
complete model Tensor schema before publishing model, optimizer, RNG, or
sequence-state changes to the caller.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

from .engine import (
    ExecutionContractError,
    SettleGraph,
    StateStore,
    _validate_state_store,
)
from .ops import AttentionState
from .plan import TypedPlan, validate_stable_id


SCHEMA_VERSION = "tide.settlegraph.checkpoint.v1"

_CHECKPOINT_ROOT_KEYS = frozenset(
    {
        "schema_version",
        "logical_plan",
        "logical_plan_hash",
        "binding",
        "typed_plan_hash",
        "model_state",
        "sequence_state",
        "optimizer_state",
        "progress",
        "training_state",
        "rng_state",
    }
)
_TORCH_OPTIMIZER_STATE_KEYS = frozenset({"state", "param_groups"})
_OPTIMIZER_RECORD_KEYS = frozenset(
    {
        "schema_version",
        "optimizer_type",
        "parameter_names",
        "state_parameter_names",
        "torch_state_dict",
    }
)
_OPTIMIZER_SCHEMA_VERSION = "tide.optimizer-state.v1"
_SUPPORTED_OPTIMIZER_TYPES = frozenset(
    {"torch.optim.adam.Adam", "torch.optim.adamw.AdamW"}
)


class CheckpointError(ValueError):
    """A checkpoint is corrupt or incompatible with the requested load."""


@dataclasses.dataclass(frozen=True)
class CheckpointArtifact:
    path: str
    sha256: str


@dataclasses.dataclass(frozen=True)
class CheckpointLoadResult:
    state: StateStore
    progress: Mapping[str, Any]
    training_state: Mapping[str, Any]
    artifact: CheckpointArtifact


def _checkpoint_stable_id(value: object, *, kind: str) -> str:
    try:
        return validate_stable_id(value, kind=kind)
    except ValueError as exc:
        raise CheckpointError(str(exc)) from exc


def _binding_dtype(typed_plan: TypedPlan, role: str) -> torch.dtype:
    """Resolve one already-validated concrete dtype role."""

    dtype_name = typed_plan.binding.dtype_roles[role]
    dtype = getattr(torch, dtype_name, None)
    if not isinstance(dtype, torch.dtype):  # Defensive against schema drift.
        raise CheckpointError(
            f"concrete binding role {role!r} has unknown dtype {dtype_name!r}"
        )
    return dtype


def _validate_model_contract(model: nn.Module, typed_plan: TypedPlan) -> SettleGraph:
    """Prove that a model and its parameters implement ``typed_plan``."""

    typed_plan.validate()
    if not isinstance(model, SettleGraph):
        raise CheckpointError("checkpoint v1 requires a SettleGraph model")
    if model.plan.canonical_dict() != typed_plan.logical_plan.canonical_dict():
        raise CheckpointError(
            "SettleGraph logical Plan does not match the checkpoint typed_plan"
        )
    if model.typed_plan is None:
        raise CheckpointError(
            "checkpoint v1 requires a SettleGraph constructed from a TypedPlan"
        )
    model.typed_plan.validate()
    if (
        model.typed_plan.logical_plan.canonical_dict()
        != typed_plan.logical_plan.canonical_dict()
        or model.typed_plan.binding.canonical_dict()
        != typed_plan.binding.canonical_dict()
        or model.typed_plan.typed_hash() != typed_plan.typed_hash()
    ):
        raise CheckpointError(
            "SettleGraph typed Plan does not match the checkpoint typed_plan"
        )

    expected_dtype = _binding_dtype(typed_plan, "parameter")
    for name, parameter in model.named_parameters():
        if not parameter.is_floating_point() or parameter.dtype != expected_dtype:
            raise CheckpointError(
                f"model parameter {name!r} dtype {parameter.dtype} does not "
                f"match binding parameter dtype {expected_dtype}"
            )
    return model


def _validate_checkpoint_state(
    model: SettleGraph, reference: Tensor, state: StateStore
) -> None:
    """Translate executor state-contract failures to checkpoint failures."""

    try:
        _validate_state_store(model, reference, state)
    except ExecutionContractError as exc:
        raise CheckpointError(f"checkpoint sequence state is invalid: {exc}") from exc


def _cpu_tensor(value: Tensor) -> Tensor:
    return value.detach().to(device="cpu").contiguous()


def _to_cpu(value: Any) -> Any:
    if isinstance(value, Tensor):
        return _cpu_tensor(value)
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def _safe_metadata(value: Any, path: str) -> Any:
    """Canonicalize the project metadata subset accepted by weights-only load."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CheckpointError(f"{path} must not contain NaN or infinity")
        return value
    if isinstance(value, Tensor):
        return _cpu_tensor(value)
    if isinstance(value, Mapping):
        normalized = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CheckpointError(f"{path} mapping keys must be strings")
            normalized[key] = _safe_metadata(item, f"{path}[{key!r}]")
        return normalized
    if isinstance(value, list):
        return [
            _safe_metadata(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, tuple):
        return tuple(
            _safe_metadata(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise CheckpointError(
        f"{path} contains unsupported weights-only type "
        f"{type(value).__name__}"
    )


def _qualified_type_name(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _finite_number(
    value: Any, path: str, *, minimum: float, strict_minimum: bool = False
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise CheckpointError(f"{path} must be a finite number")
    numeric = float(value)
    invalid = numeric <= minimum if strict_minimum else numeric < minimum
    if invalid:
        relation = ">" if strict_minimum else ">="
        raise CheckpointError(f"{path} must satisfy value {relation} {minimum}")
    return numeric


def _validate_optimizer_group_options(
    optimizer_type: str, param_groups: Any
) -> None:
    """Validate the formula-bearing Adam/AdamW hyperparameter domain."""

    if optimizer_type not in _SUPPORTED_OPTIMIZER_TYPES:
        raise CheckpointError(
            f"checkpoint v1 does not support optimizer type {optimizer_type!r}"
        )
    if not isinstance(param_groups, list):
        raise CheckpointError("optimizer param_groups must be a list")
    for group_index, group in enumerate(param_groups):
        if not isinstance(group, Mapping):
            raise CheckpointError(
                f"optimizer param_group {group_index} must be a mapping"
            )
        prefix = f"optimizer_state.param_groups[{group_index}]"
        _finite_number(group.get("lr"), f"{prefix}['lr']", minimum=0.0)
        _finite_number(
            group.get("eps"),
            f"{prefix}['eps']",
            minimum=0.0,
            strict_minimum=True,
        )
        _finite_number(
            group.get("weight_decay"),
            f"{prefix}['weight_decay']",
            minimum=0.0,
        )
        betas = group.get("betas")
        if not isinstance(betas, tuple) or len(betas) != 2:
            raise CheckpointError(f"{prefix}['betas'] must be a pair")
        for beta_index, beta in enumerate(betas):
            numeric = _finite_number(
                beta,
                f"{prefix}['betas'][{beta_index}]",
                minimum=0.0,
            )
            if numeric >= 1.0:
                raise CheckpointError(
                    f"{prefix}['betas'][{beta_index}] must be less than 1"
                )
        for key in (
            "amsgrad",
            "maximize",
            "capturable",
            "differentiable",
            "decoupled_weight_decay",
        ):
            if type(group.get(key)) is not bool:
                raise CheckpointError(f"{prefix}[{key!r}] must be boolean")
        for key in ("foreach", "fused"):
            if group.get(key) is not None and type(group.get(key)) is not bool:
                raise CheckpointError(
                    f"{prefix}[{key!r}] must be boolean or null"
                )
        if "initial_lr" in group:
            _finite_number(
                group["initial_lr"],
                f"{prefix}['initial_lr']",
                minimum=0.0,
            )


def _optimizer_parameter_name_groups(
    model: nn.Module, optimizer: torch.optim.Optimizer
) -> Tuple[Tuple[str, ...], ...]:
    """Bind every live optimizer parameter to one stable model key."""

    model_name_by_identity = {
        id(parameter): name for name, parameter in model.named_parameters()
    }
    seen_parameters = set()
    name_groups = []
    for group_index, group in enumerate(optimizer.param_groups):
        if not isinstance(group, Mapping):
            raise CheckpointError(
                f"optimizer parameter group {group_index} must be a mapping"
            )
        parameters = group.get("params")
        if not isinstance(parameters, list):
            raise CheckpointError(
                f"optimizer parameter group {group_index} params must be a list"
            )
        names = []
        for parameter_index, parameter in enumerate(parameters):
            if not isinstance(parameter, Tensor):
                raise CheckpointError(
                    "optimizer parameter entries must be model Tensors"
                )
            identity = id(parameter)
            name = model_name_by_identity.get(identity)
            if name is None:
                raise CheckpointError(
                    "optimizer parameter "
                    f"{group_index}:{parameter_index} is not owned by the "
                    "checkpoint model"
                )
            if identity in seen_parameters:
                raise CheckpointError(
                    f"optimizer parameter {name!r} appears more than once"
                )
            seen_parameters.add(identity)
            names.append(name)
        name_groups.append(tuple(names))
    return tuple(name_groups)


def _walk_optimizer_tensors(
    value: Any, path: str
) -> Iterable[Tuple[str, Tensor]]:
    if isinstance(value, Tensor):
        yield path, value
        return
    if isinstance(value, Mapping):
        for key in sorted(value, key=repr):
            yield from _walk_optimizer_tensors(
                value[key], f"{path}[{key!r}]"
            )
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _walk_optimizer_tensors(item, f"{path}[{index}]")


def _validate_optimizer_tensor_storage(
    state: Any, *, require_cpu: bool
) -> None:
    """Reject mutable optimizer Tensor aliases, including distinct views."""

    storage_owners: Dict[Tuple[str, Optional[int], int], str] = {}
    storage_ranges = []
    for path, tensor in _walk_optimizer_tensors(
        state, "optimizer_state.torch_state_dict.state"
    ):
        if tensor.layout != torch.strided:
            raise CheckpointError(
                f"{path} uses unsupported optimizer Tensor layout {tensor.layout}"
            )
        if require_cpu and tensor.device.type != "cpu":
            raise CheckpointError(f"{path} must be stored on CPU")
        storage = tensor.untyped_storage()
        storage_id = (
            tensor.device.type,
            tensor.device.index,
            storage._cdata,
        )
        owner = storage_owners.get(storage_id)
        if owner is not None and owner != path:
            raise CheckpointError(
                f"mutable optimizer storage is shared by {owner} and {path}"
            )
        start = storage.data_ptr()
        end = start + storage.nbytes()
        if start and end > start:
            for (
                other_type,
                other_index,
                other_start,
                other_end,
                other_path,
            ) in storage_ranges:
                if (
                    other_type == tensor.device.type
                    and other_index == tensor.device.index
                    and start < other_end
                    and other_start < end
                ):
                    raise CheckpointError(
                        "mutable optimizer storage is shared by "
                        f"{other_path} and {path}"
                    )
            storage_ranges.append(
                (
                    tensor.device.type,
                    tensor.device.index,
                    start,
                    end,
                    path,
                )
            )
        storage_owners[storage_id] = path


def _build_optimizer_record(
    model: nn.Module, optimizer: torch.optim.Optimizer
) -> Dict[str, Any]:
    """Create a CPU optimizer record with stable model-parameter ownership."""

    name_groups = _optimizer_parameter_name_groups(model, optimizer)
    torch_state = optimizer.state_dict()
    if (
        not isinstance(torch_state, Mapping)
        or set(torch_state) != _TORCH_OPTIMIZER_STATE_KEYS
        or not isinstance(torch_state["state"], Mapping)
        or not isinstance(torch_state["param_groups"], list)
        or len(torch_state["param_groups"]) != len(name_groups)
    ):
        raise CheckpointError("optimizer exposes an unsupported state schema")
    optimizer_type = _qualified_type_name(optimizer)
    _validate_optimizer_group_options(
        optimizer_type, torch_state["param_groups"]
    )

    name_by_parameter_id: Dict[int, str] = {}
    for group_index, (group, names) in enumerate(
        zip(torch_state["param_groups"], name_groups)
    ):
        if not isinstance(group, Mapping) or not isinstance(
            group.get("params"), list
        ):
            raise CheckpointError(
                f"optimizer serialized group {group_index} is malformed"
            )
        parameter_ids = group["params"]
        if len(parameter_ids) != len(names):
            raise CheckpointError(
                f"optimizer serialized group {group_index} changed parameter order"
            )
        for parameter_id, name in zip(parameter_ids, names):
            if (
                type(parameter_id) is not int
                or parameter_id in name_by_parameter_id
            ):
                raise CheckpointError(
                    "optimizer serialized parameter identifiers must be unique "
                    "integers"
                )
            name_by_parameter_id[parameter_id] = name

    state_parameter_names = []
    for parameter_id in torch_state["state"]:
        if type(parameter_id) is not int or parameter_id not in name_by_parameter_id:
            raise CheckpointError(
                "optimizer state contains an unmapped parameter identifier"
            )
        state_parameter_names.append(name_by_parameter_id[parameter_id])
    _validate_optimizer_tensor_storage(torch_state["state"], require_cpu=False)
    return {
        "schema_version": _OPTIMIZER_SCHEMA_VERSION,
        "optimizer_type": optimizer_type,
        "parameter_names": [list(names) for names in name_groups],
        "state_parameter_names": sorted(state_parameter_names),
        "torch_state_dict": _to_cpu(torch_state),
    }


def _parse_optimizer_record(
    model: nn.Module, record: Any
) -> Tuple[Mapping[str, Any], Tuple[Tuple[str, ...], ...], str]:
    """Validate project metadata before consulting a target optimizer."""

    if not isinstance(record, Mapping) or set(record) != _OPTIMIZER_RECORD_KEYS:
        raise CheckpointError(
            "optimizer_state has an unexpected project record key set"
        )
    if record["schema_version"] != _OPTIMIZER_SCHEMA_VERSION:
        raise CheckpointError("optimizer_state schema version is incompatible")
    optimizer_type = record["optimizer_type"]
    if not isinstance(optimizer_type, str) or not optimizer_type:
        raise CheckpointError("optimizer_state optimizer_type must be a string")

    raw_name_groups = record["parameter_names"]
    if not isinstance(raw_name_groups, list):
        raise CheckpointError("optimizer_state parameter_names must be a list")
    known_model_names = {name for name, _ in model.named_parameters()}
    seen_names = set()
    name_groups = []
    for group_index, raw_names in enumerate(raw_name_groups):
        if not isinstance(raw_names, list) or not all(
            isinstance(name, str) for name in raw_names
        ):
            raise CheckpointError(
                f"optimizer_state parameter_names group {group_index} is invalid"
            )
        names = tuple(raw_names)
        for name in names:
            if name not in known_model_names or name in seen_names:
                raise CheckpointError(
                    "optimizer_state parameter names must uniquely identify "
                    "checkpoint model parameters"
                )
            seen_names.add(name)
        name_groups.append(names)

    torch_state = record["torch_state_dict"]
    if (
        not isinstance(torch_state, Mapping)
        or set(torch_state) != _TORCH_OPTIMIZER_STATE_KEYS
        or not isinstance(torch_state["state"], Mapping)
        or not isinstance(torch_state["param_groups"], list)
        or len(torch_state["param_groups"]) != len(name_groups)
    ):
        raise CheckpointError(
            "optimizer_state torch_state_dict must contain exactly state and "
            "compatible param_groups"
        )
    _validate_optimizer_group_options(
        optimizer_type, torch_state["param_groups"]
    )

    name_by_parameter_id: Dict[int, str] = {}
    for group_index, (group, names) in enumerate(
        zip(torch_state["param_groups"], name_groups)
    ):
        if not isinstance(group, Mapping) or not isinstance(
            group.get("params"), list
        ):
            raise CheckpointError(
                f"optimizer_state param_group {group_index} is invalid"
            )
        parameter_ids = group["params"]
        if len(parameter_ids) != len(names):
            raise CheckpointError(
                f"optimizer_state param_group {group_index} mapping is incompatible"
            )
        for parameter_id, name in zip(parameter_ids, names):
            if (
                type(parameter_id) is not int
                or parameter_id in name_by_parameter_id
            ):
                raise CheckpointError(
                    "optimizer_state parameter identifiers must be unique integers"
                )
            name_by_parameter_id[parameter_id] = name

    mapped_state_names = []
    for parameter_id in torch_state["state"]:
        if type(parameter_id) is not int or parameter_id not in name_by_parameter_id:
            raise CheckpointError(
                "optimizer state contains an unmapped parameter identifier"
            )
        mapped_state_names.append(name_by_parameter_id[parameter_id])
    state_parameter_names = record["state_parameter_names"]
    if (
        not isinstance(state_parameter_names, list)
        or not all(isinstance(name, str) for name in state_parameter_names)
        or state_parameter_names != sorted(set(state_parameter_names))
        or state_parameter_names != sorted(mapped_state_names)
    ):
        raise CheckpointError(
            "optimizer_state initialized-state parameter manifest does not "
            "match its state mapping"
        )
    _validate_optimizer_tensor_storage(torch_state["state"], require_cpu=True)
    return torch_state, tuple(name_groups), optimizer_type


def serialize_state_store(store: StateStore) -> Dict[str, Any]:
    """Return the canonical CPU representation of sequence state."""

    for key in store.values:
        if not isinstance(key, tuple) or len(key) != 2:
            raise CheckpointError(
                "receiver-state keys must be (sequence_id, node_id) pairs"
            )
        _checkpoint_stable_id(key[0], kind="state sequence")
        _checkpoint_stable_id(key[1], kind="state node")
    for key in store.selector_history:
        if not isinstance(key, tuple) or len(key) != 2:
            raise CheckpointError(
                "selector-history keys must be (sequence_id, owner_id) pairs"
            )
        _checkpoint_stable_id(key[0], kind="state sequence")
        _checkpoint_stable_id(key[1], kind="selector-history owner")
    for sequence_id, position in store.next_position.items():
        _checkpoint_stable_id(sequence_id, kind="state sequence")
        if type(position) is not int or position < 0:
            raise CheckpointError(
                "next Token positions must be nonnegative integers"
            )

    values = []
    for (sequence_id, node_id), state in sorted(store.values.items()):
        if state is None:
            payload = {"kind": "none"}
        elif isinstance(state, AttentionState):
            payload = {
                "kind": "attention_window",
                "positions": _cpu_tensor(state.positions),
                "keys": _cpu_tensor(state.keys),
                "values": _cpu_tensor(state.values),
            }
        elif isinstance(state, Tensor):
            payload = {"kind": "tensor", "value": _cpu_tensor(state)}
        else:
            raise CheckpointError(
                f"unsupported receiver state type {type(state).__name__}"
            )
        values.append(
            {
                "sequence_id": sequence_id,
                "node_id": node_id,
                "payload": payload,
            }
        )
    history = []
    for (sequence_id, owner_id), value in sorted(store.selector_history.items()):
        if not isinstance(value, Tensor):
            raise CheckpointError(
                "checkpoint v1 only serializes Tensor selector-history values"
            )
        history.append(
            {
                "sequence_id": sequence_id,
                "owner_id": owner_id,
                "value": _cpu_tensor(value),
            }
        )
    return {
        "receiver_values": values,
        "selector_history": history,
        "next_position": dict(sorted(store.next_position.items())),
    }


def deserialize_state_store(
    record: Mapping[str, Any],
    *,
    device: Union[str, torch.device],
    dtype: torch.dtype,
) -> StateStore:
    """Restore canonical state without attaching it to an executor yet."""

    if not isinstance(record, Mapping):
        raise CheckpointError("sequence-state record must be a mapping")
    if set(record) != {"receiver_values", "selector_history", "next_position"}:
        raise CheckpointError("sequence-state record has an unexpected key set")
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise CheckpointError("sequence-state dtype must be a floating torch dtype")
    raw_positions = record["next_position"]
    if not isinstance(raw_positions, Mapping):
        raise CheckpointError("next_position must be a mapping")
    next_position: Dict[str, int] = {}
    for sequence_id, position in raw_positions.items():
        validated_sequence_id = _checkpoint_stable_id(
            sequence_id, kind="state sequence"
        )
        if type(position) is not int or position < 0:
            raise CheckpointError("invalid sequence position record")
        next_position[validated_sequence_id] = position

    raw_values = record["receiver_values"]
    if not isinstance(raw_values, (list, tuple)):
        raise CheckpointError("receiver_values must be a sequence")
    validated_values = []
    value_keys = set()
    for item in raw_values:
        if not isinstance(item, Mapping) or set(item) != {
            "sequence_id",
            "node_id",
            "payload",
        }:
            raise CheckpointError("invalid receiver-state entry")
        key = (
            _checkpoint_stable_id(
                item["sequence_id"], kind="state sequence"
            ),
            _checkpoint_stable_id(item["node_id"], kind="state node"),
        )
        if key in value_keys:
            raise CheckpointError("receiver-state keys must be unique")
        value_keys.add(key)
        payload = item["payload"]
        if not isinstance(payload, Mapping):
            raise CheckpointError("receiver-state payload must be a mapping")
        kind = payload.get("kind")
        if kind == "none":
            if set(payload) != {"kind"}:
                raise CheckpointError("none state payload has an unexpected key set")
        elif kind == "tensor":
            if set(payload) != {"kind", "value"}:
                raise CheckpointError("Tensor state payload has an unexpected key set")
            value = payload.get("value")
            if (
                not isinstance(value, Tensor)
                or not value.is_floating_point()
                or value.dtype != dtype
            ):
                raise CheckpointError(
                    "Tensor state payload must use the binding state dtype"
                )
        elif kind == "attention_window":
            if set(payload) != {"kind", "positions", "keys", "values"}:
                raise CheckpointError(
                    "Attention state payload has an unexpected key set"
                )
            positions = payload.get("positions")
            keys = payload.get("keys")
            state_values = payload.get("values")
            if not all(
                isinstance(value, Tensor)
                for value in (positions, keys, state_values)
            ):
                raise CheckpointError("Attention state payload is invalid")
            if positions.dtype != torch.int64:
                raise CheckpointError("Attention positions must already use int64")
            if any(
                not value.is_floating_point() or value.dtype != dtype
                for value in (keys, state_values)
            ):
                raise CheckpointError(
                    "Attention key/value payloads must use the binding state dtype"
                )
        else:
            raise CheckpointError(f"unknown receiver-state kind {kind!r}")
        validated_values.append((key, kind, payload))

    raw_history = record["selector_history"]
    if not isinstance(raw_history, (list, tuple)):
        raise CheckpointError("selector_history must be a sequence")
    validated_history = []
    history_keys = set()
    for item in raw_history:
        if not isinstance(item, Mapping) or set(item) != {
            "sequence_id",
            "owner_id",
            "value",
        }:
            raise CheckpointError("invalid selector-history entry")
        key = (
            _checkpoint_stable_id(
                item["sequence_id"], kind="state sequence"
            ),
            _checkpoint_stable_id(
                item["owner_id"], kind="selector-history owner"
            ),
        )
        value = item["value"]
        if (
            key in history_keys
            or not isinstance(value, Tensor)
            or not value.is_floating_point()
            or value.dtype != dtype
        ):
            raise CheckpointError(
                "invalid selector-history key or binding state dtype"
            )
        history_keys.add(key)
        validated_history.append((key, value))

    # All IDs and CPU payload schemas are known-good before the first device
    # transfer.  This keeps malformed checkpoint text from causing partial
    # accelerator work before the loader rejects it.
    values = {}
    for key, kind, payload in validated_values:
        if kind == "none":
            state = None
        elif kind == "tensor":
            state = payload["value"].to(device=device)
        else:
            state = AttentionState(
                payload["positions"].to(device=device),
                payload["keys"].to(device=device),
                payload["values"].to(device=device),
            )
        values[key] = state
    history = {
        key: value.to(device=device) for key, value in validated_history
    }
    return StateStore(values, history, next_position)


def _file_sha256(path: Union[str, os.PathLike[str]]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(
    path: Union[str, os.PathLike[str]],
    *,
    model: nn.Module,
    typed_plan: TypedPlan,
    state: StateStore,
    optimizer: Optional[torch.optim.Optimizer] = None,
    progress: Optional[Mapping[str, Any]] = None,
    training_state: Optional[Mapping[str, Any]] = None,
) -> CheckpointArtifact:
    """Atomically save a CPU checkpoint and return its content digest."""

    model = _validate_model_contract(model, typed_plan)
    anchor = next(model.parameters(), None)
    if anchor is None:
        raise CheckpointError("checkpoint target model has no dtype/device anchor")
    _validate_checkpoint_state(
        model,
        anchor.new_empty((1, typed_plan.logical_plan.d_model)),
        state,
    )
    if progress is not None and not isinstance(progress, Mapping):
        raise CheckpointError("progress must be a mapping")
    if training_state is not None and not isinstance(training_state, Mapping):
        raise CheckpointError("training_state must be a mapping")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "logical_plan": typed_plan.logical_plan.canonical_dict(),
        "logical_plan_hash": typed_plan.logical_hash(),
        "binding": typed_plan.binding.canonical_dict(),
        "typed_plan_hash": typed_plan.typed_hash(),
        "model_state": _to_cpu(model.state_dict()),
        "sequence_state": serialize_state_store(state),
        "optimizer_state": (
            _build_optimizer_record(model, optimizer)
            if optimizer is not None
            else None
        ),
        "progress": _safe_metadata(dict(progress or {}), "progress"),
        "training_state": _safe_metadata(
            dict(training_state or {}), "training_state"
        ),
        "rng_state": {"torch_cpu": torch.get_rng_state().clone()},
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=str(destination.parent)
    )
    os.close(handle)
    try:
        torch.save(payload, temporary)
        try:
            probe = torch.load(
                temporary, map_location="cpu", weights_only=True
            )
        except Exception as exc:
            raise CheckpointError(
                "saved checkpoint failed its weights-only self-check: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if (
            not isinstance(probe, Mapping)
            or set(probe) != _CHECKPOINT_ROOT_KEYS
            or probe.get("schema_version") != SCHEMA_VERSION
            or probe.get("logical_plan_hash") != typed_plan.logical_hash()
            or probe.get("typed_plan_hash") != typed_plan.typed_hash()
        ):
            raise CheckpointError(
                "saved checkpoint failed its root identity self-check"
            )
        os.replace(temporary, destination)
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise
    return CheckpointArtifact(str(destination), _file_sha256(destination))


def _load_payload(
    path: Union[str, os.PathLike[str]], expected_sha256: Optional[str]
) -> Tuple[Mapping[str, Any], CheckpointArtifact]:
    actual_hash = _file_sha256(path)
    if expected_sha256 is not None and actual_hash != expected_sha256:
        raise CheckpointError(
            f"checkpoint SHA-256 mismatch: expected {expected_sha256}, got {actual_hash}"
        )
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise CheckpointError(
            f"cannot load checkpoint: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise CheckpointError("checkpoint root must be a mapping")
    return payload, CheckpointArtifact(str(path), actual_hash)


def _validate_model_state(model: nn.Module, saved: Any) -> None:
    if not isinstance(saved, Mapping):
        raise CheckpointError("model_state must be a mapping")
    current = model.state_dict()
    if set(saved) != set(current):
        missing = sorted(set(current) - set(saved))
        extra = sorted(set(saved) - set(current))
        raise CheckpointError(
            f"model key mismatch; missing={missing!r}, extra={extra!r}"
        )
    for key, reference in current.items():
        value = saved[key]
        if not isinstance(value, Tensor):
            raise CheckpointError(f"model state {key!r} is not a Tensor")
        if value.shape != reference.shape or value.dtype != reference.dtype:
            raise CheckpointError(
                f"model state {key!r} shape/dtype mismatch: saved "
                f"{tuple(value.shape)}/{value.dtype}, expected "
                f"{tuple(reference.shape)}/{reference.dtype}"
            )


def _validate_root_key_set(payload: Mapping[str, Any]) -> None:
    actual = set(payload)
    if actual == _CHECKPOINT_ROOT_KEYS:
        return
    missing = sorted(_CHECKPOINT_ROOT_KEYS - actual)
    extra = sorted(actual - _CHECKPOINT_ROOT_KEYS, key=repr)
    raise CheckpointError(
        "checkpoint root has an unexpected key set; "
        f"missing={missing!r}, extra={extra!r}"
    )


def _validate_cpu_rng_state(record: Any) -> Tensor:
    """Validate CPU RNG bytes without mutating the process-global generator."""

    if not isinstance(record, Mapping) or set(record) != {"torch_cpu"}:
        raise CheckpointError("checkpoint CPU RNG record has an unexpected key set")
    state = record["torch_cpu"]
    if (
        not isinstance(state, Tensor)
        or state.dtype != torch.uint8
        or state.device.type != "cpu"
        or state.layout != torch.strided
        or state.ndim != 1
        or not state.is_contiguous()
    ):
        raise CheckpointError(
            "checkpoint CPU RNG state must be a contiguous one-dimensional "
            "CPU uint8 Tensor"
        )
    try:
        probe = torch.Generator(device="cpu")
        probe.set_state(state.detach().clone())
    except Exception as exc:
        raise CheckpointError(
            "checkpoint CPU RNG state cannot be restored: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return state


def _validate_optimizer_value(saved: Any, reference: Any, path: str) -> None:
    """Compare one optimizer value against a locally generated schema."""

    if isinstance(saved, Tensor):
        if not isinstance(reference, Tensor):
            raise CheckpointError(
                f"{path} is a Tensor but the target optimizer expects "
                f"{type(reference).__name__}"
            )
        if saved.device.type != "cpu":
            raise CheckpointError(f"{path} must be stored on CPU")
        if (
            saved.shape != reference.shape
            or saved.dtype != reference.dtype
            or saved.layout != reference.layout
        ):
            raise CheckpointError(
                f"{path} Tensor shape/dtype/layout mismatch: saved "
                f"{tuple(saved.shape)}/{saved.dtype}/{saved.layout}, expected "
                f"{tuple(reference.shape)}/{reference.dtype}/{reference.layout}"
            )
        return
    if isinstance(reference, Tensor):
        raise CheckpointError(f"{path} must be a Tensor")

    if isinstance(saved, Mapping):
        if not isinstance(reference, Mapping):
            raise CheckpointError(
                f"{path} is a mapping but the target optimizer expects "
                f"{type(reference).__name__}"
            )
        if set(saved) != set(reference):
            missing = sorted(set(reference) - set(saved), key=repr)
            extra = sorted(set(saved) - set(reference), key=repr)
            raise CheckpointError(
                f"{path} has an unexpected key set; "
                f"missing={missing!r}, extra={extra!r}"
            )
        for key in reference:
            _validate_optimizer_value(
                saved[key], reference[key], f"{path}[{key!r}]"
            )
        return

    if isinstance(saved, (list, tuple)):
        if type(saved) is not type(reference) or len(saved) != len(reference):
            reference_length = (
                len(reference) if isinstance(reference, (list, tuple)) else "n/a"
            )
            raise CheckpointError(
                f"{path} sequence type/length mismatch: saved "
                f"{type(saved).__name__}/{len(saved)}, expected "
                f"{type(reference).__name__}/{reference_length}"
            )
        for index, (saved_item, reference_item) in enumerate(
            zip(saved, reference)
        ):
            _validate_optimizer_value(
                saved_item, reference_item, f"{path}[{index}]"
            )
        return

    if type(saved) is not type(reference):
        raise CheckpointError(
            f"{path} type mismatch: saved {type(saved).__name__}, expected "
            f"{type(reference).__name__}"
        )


def _make_scratch_optimizer(
    optimizer: torch.optim.Optimizer,
) -> torch.optim.Optimizer:
    """Deep-copy an optimizer and prove that its parameters are independent."""

    try:
        scratch = copy.deepcopy(optimizer)
    except Exception as exc:
        raise CheckpointError(
            "cannot construct an independent optimizer for checkpoint "
            f"validation: {type(exc).__name__}: {exc}"
        ) from exc
    if type(scratch) is not type(optimizer):
        raise CheckpointError(
            "independent optimizer validation changed the optimizer type"
        )
    if len(scratch.param_groups) != len(optimizer.param_groups):
        raise CheckpointError(
            "independent optimizer validation changed the parameter groups"
        )
    for group_index, (live_group, scratch_group) in enumerate(
        zip(optimizer.param_groups, scratch.param_groups)
    ):
        live_parameters = live_group.get("params")
        scratch_parameters = scratch_group.get("params")
        if (
            not isinstance(live_parameters, list)
            or not isinstance(scratch_parameters, list)
            or len(live_parameters) != len(scratch_parameters)
        ):
            raise CheckpointError(
                f"optimizer parameter group {group_index} is malformed"
            )
        for parameter_index, (live, candidate) in enumerate(
            zip(live_parameters, scratch_parameters)
        ):
            if (
                not isinstance(live, Tensor)
                or not isinstance(candidate, Tensor)
                or candidate is live
                or candidate.shape != live.shape
                or candidate.dtype != live.dtype
                or candidate.device != live.device
            ):
                raise CheckpointError(
                    "independent optimizer validation did not clone parameter "
                    f"{group_index}:{parameter_index} safely"
                )
    return scratch


def _validate_optimizer_state(
    optimizer: torch.optim.Optimizer, saved: Any
) -> None:
    """Strictly validate serialized optimizer mapping and Tensor schemas."""

    if (
        not isinstance(saved, Mapping)
        or set(saved) != _TORCH_OPTIMIZER_STATE_KEYS
    ):
        raise CheckpointError(
            "optimizer_state must be a mapping with exactly state and "
            "param_groups"
        )
    saved_state = saved["state"]
    saved_groups = saved["param_groups"]
    if not isinstance(saved_state, Mapping) or not isinstance(saved_groups, list):
        raise CheckpointError(
            "optimizer_state state/param_groups containers are invalid"
        )

    current = optimizer.state_dict()
    if (
        not isinstance(current, Mapping)
        or set(current) != _TORCH_OPTIMIZER_STATE_KEYS
    ):
        raise CheckpointError("target optimizer exposes an unsupported state schema")
    current_state = current["state"]
    current_groups = current["param_groups"]
    if (
        not isinstance(current_state, Mapping)
        or not isinstance(current_groups, list)
        or len(saved_groups) != len(current_groups)
        or len(saved_groups) != len(optimizer.param_groups)
    ):
        raise CheckpointError("optimizer parameter-group mapping is incompatible")

    parameter_by_id: Dict[int, Tensor] = {}
    for group_index, (saved_group, current_group, live_group) in enumerate(
        zip(saved_groups, current_groups, optimizer.param_groups)
    ):
        if not all(
            isinstance(group, Mapping)
            for group in (saved_group, current_group, live_group)
        ):
            raise CheckpointError(
                f"optimizer parameter group {group_index} must be a mapping"
            )
        if set(saved_group) != set(current_group):
            raise CheckpointError(
                f"optimizer parameter group {group_index} has an unexpected key set"
            )
        saved_parameters = saved_group.get("params")
        current_parameters = current_group.get("params")
        live_parameters = live_group.get("params")
        if (
            not isinstance(saved_parameters, list)
            or not isinstance(current_parameters, list)
            or not isinstance(live_parameters, list)
            or len(saved_parameters) != len(current_parameters)
            or len(saved_parameters) != len(live_parameters)
        ):
            raise CheckpointError(
                f"optimizer parameter group {group_index} mapping is incompatible"
            )
        for parameter_id, current_id, parameter in zip(
            saved_parameters, current_parameters, live_parameters
        ):
            if (
                type(parameter_id) is not int
                or type(current_id) is not int
                or parameter_id != current_id
                or parameter_id in parameter_by_id
                or not isinstance(parameter, Tensor)
            ):
                raise CheckpointError(
                    "optimizer parameter identifiers must uniquely map to target "
                    "parameters"
                )
            parameter_by_id[parameter_id] = parameter
        for key in current_group:
            if key != "params":
                _validate_optimizer_value(
                    saved_group[key],
                    current_group[key],
                    f"optimizer_state.param_groups[{group_index}][{key!r}]",
                )

    for parameter_id in saved_state:
        if type(parameter_id) is not int or parameter_id not in parameter_by_id:
            raise CheckpointError(
                "optimizer state contains an unmapped parameter identifier"
            )

    global_rng = torch.get_rng_state().clone()
    try:
        scratch = _make_scratch_optimizer(optimizer)
        reference_state: Mapping[int, Any] = {}
        if saved_state:
            # Do not trust pre-existing target state as the schema oracle.  A
            # clean, independent first step establishes the optimizer's own
            # state keys and Tensor metadata for every mapped parameter.
            scratch.state.clear()
            for group in scratch.param_groups:
                for parameter in group["params"]:
                    parameter.grad = torch.zeros_like(
                        parameter, memory_format=torch.preserve_format
                    )
            try:
                scratch.step()
            except Exception as exc:
                raise CheckpointError(
                    "cannot establish the target optimizer Tensor schema safely: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            reference_state = scratch.state_dict()["state"]

        for parameter_id, value in saved_state.items():
            if parameter_id not in reference_state:
                raise CheckpointError(
                    "target optimizer did not establish state for parameter "
                    f"identifier {parameter_id}"
                )
            _validate_optimizer_value(
                value,
                reference_state[parameter_id],
                f"optimizer_state.state[{parameter_id}]",
            )

        for parameter_id, value in saved_state.items():
            for key, tensor in value.items():
                if isinstance(tensor, Tensor) and not bool(
                    torch.isfinite(tensor).all().item()
                ):
                    raise CheckpointError(
                        "optimizer state Tensor must be finite: "
                        f"state[{parameter_id}][{key!r}]"
                    )
            step = value.get("step")
            if isinstance(step, Tensor):
                step_value = float(step.item())
                if step_value < 0.0 or not step_value.is_integer():
                    raise CheckpointError(
                        f"optimizer state[{parameter_id}] step must be a "
                        "nonnegative integer value"
                    )
            for key in ("exp_avg_sq", "max_exp_avg_sq"):
                moment = value.get(key)
                if isinstance(moment, Tensor) and bool((moment < 0).any().item()):
                    raise CheckpointError(
                        f"optimizer state[{parameter_id}][{key!r}] must be "
                        "nonnegative"
                    )

        # Exercise PyTorch's own loader only on an independent optimizer.  The
        # manual checks above run first because the loader may silently accept
        # or cast malformed Tensor shapes and dtypes.
        try:
            scratch.load_state_dict(copy.deepcopy(saved))
        except Exception as exc:
            raise CheckpointError(
                "optimizer state is incompatible: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
    finally:
        torch.set_rng_state(global_rng)


@dataclasses.dataclass
class _OptimizerSnapshot:
    state: Any
    param_groups: Any
    defaults: Any


def _snapshot_optimizer(
    optimizer: torch.optim.Optimizer,
) -> _OptimizerSnapshot:
    if isinstance(optimizer.state, defaultdict):
        state = defaultdict(optimizer.state.default_factory)
    else:
        try:
            state = type(optimizer.state)()
        except Exception:
            state = {}
    for parameter, value in optimizer.state.items():
        state[parameter] = copy.deepcopy(value)
    param_groups = []
    for group in optimizer.param_groups:
        param_groups.append(
            {
                key: list(value) if key == "params" else copy.deepcopy(value)
                for key, value in group.items()
            }
        )
    return _OptimizerSnapshot(
        state=state,
        param_groups=param_groups,
        defaults=copy.deepcopy(optimizer.defaults),
    )


def _restore_optimizer(
    optimizer: torch.optim.Optimizer, snapshot: _OptimizerSnapshot
) -> None:
    optimizer.state = snapshot.state
    optimizer.param_groups = snapshot.param_groups
    optimizer.defaults = snapshot.defaults


def _snapshot_model(model: nn.Module) -> Dict[str, Tensor]:
    return {
        key: value.detach().clone(memory_format=torch.preserve_format)
        for key, value in model.state_dict().items()
    }


def _restore_model(model: nn.Module, snapshot: Mapping[str, Tensor]) -> None:
    current = model.state_dict()
    if set(current) != set(snapshot):
        raise RuntimeError("model state keys changed during checkpoint commit")
    with torch.no_grad():
        for key, value in current.items():
            value.copy_(snapshot[key])


def load_checkpoint(
    path: Union[str, os.PathLike[str]],
    *,
    model: nn.Module,
    typed_plan: TypedPlan,
    mode: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    expected_sha256: Optional[str] = None,
) -> CheckpointLoadResult:
    """Validate then apply an ``init-from`` or full ``resume`` load."""

    if mode not in {"init-from", "resume"}:
        raise CheckpointError("mode must be 'init-from' or 'resume'")
    model = _validate_model_contract(model, typed_plan)
    payload, artifact = _load_payload(path, expected_sha256)
    _validate_root_key_set(payload)
    if payload["schema_version"] != SCHEMA_VERSION:
        raise CheckpointError("checkpoint schema version is incompatible")
    if payload["logical_plan_hash"] != typed_plan.logical_hash():
        raise CheckpointError("checkpoint logical Plan hash does not match")
    if payload["logical_plan"] != typed_plan.logical_plan.canonical_dict():
        raise CheckpointError("checkpoint normalized logical Plan does not match")
    if payload["typed_plan_hash"] != typed_plan.typed_hash():
        raise CheckpointError("checkpoint typed Plan hash does not match")
    if payload["binding"] != typed_plan.binding.canonical_dict():
        raise CheckpointError("checkpoint concrete dtype binding does not match")
    _validate_model_state(model, payload["model_state"])

    parameter = next(model.parameters(), None)
    if parameter is None:
        raise CheckpointError("checkpoint target model has no dtype/device anchor")
    state_dtype = _binding_dtype(typed_plan, "state")
    decoded_cpu_state = deserialize_state_store(
        payload["sequence_state"],
        device="cpu",
        dtype=state_dtype,
    )
    _validate_checkpoint_state(
        model,
        torch.empty(
            (1, typed_plan.logical_plan.d_model),
            device="cpu",
            dtype=state_dtype,
        ),
        decoded_cpu_state,
    )
    if parameter.device.type == "cpu":
        decoded_state = decoded_cpu_state
    else:
        decoded_state = deserialize_state_store(
            payload["sequence_state"],
            device=parameter.device,
            dtype=state_dtype,
        )
        _validate_checkpoint_state(
            model,
            parameter.new_empty((1, typed_plan.logical_plan.d_model)),
            decoded_state,
        )
    progress_record = payload["progress"]
    training_state_record = payload["training_state"]
    if not isinstance(progress_record, Mapping) or not isinstance(
        training_state_record, Mapping
    ):
        raise CheckpointError("progress and training_state must be mappings")
    progress_record = _safe_metadata(progress_record, "progress")
    training_state_record = _safe_metadata(
        training_state_record, "training_state"
    )
    validated_rng_state = _validate_cpu_rng_state(payload["rng_state"])
    saved_optimizer_record = payload["optimizer_state"]
    saved_optimizer_state: Optional[Mapping[str, Any]] = None
    saved_optimizer_name_groups: Tuple[Tuple[str, ...], ...] = ()
    saved_optimizer_type: Optional[str] = None
    if saved_optimizer_record is not None:
        (
            saved_optimizer_state,
            saved_optimizer_name_groups,
            saved_optimizer_type,
        ) = _parse_optimizer_record(model, saved_optimizer_record)

    if mode == "resume":
        state = decoded_state
        if (optimizer is None) != (saved_optimizer_record is None):
            raise CheckpointError(
                "resume requires the same optimizer-presence contract as the checkpoint"
            )
        if optimizer is not None:
            if _qualified_type_name(optimizer) != saved_optimizer_type:
                raise CheckpointError(
                    "target optimizer type does not match the checkpoint: "
                    f"{_qualified_type_name(optimizer)!r} != "
                    f"{saved_optimizer_type!r}"
                )
            target_name_groups = _optimizer_parameter_name_groups(
                model, optimizer
            )
            if target_name_groups != saved_optimizer_name_groups:
                raise CheckpointError(
                    "target optimizer parameter-to-model mapping does not "
                    "match the checkpoint"
                )
            assert saved_optimizer_state is not None
            _validate_optimizer_state(optimizer, saved_optimizer_state)
        progress = progress_record
        training_state = training_state_record
    else:
        state = StateStore()
        progress = {}
        training_state = {}

    # Validation above is side-effect free with respect to the caller.  Treat
    # the remaining mutations as one transaction and restore all three mutable
    # participants even if a loader mutates before raising.
    try:
        model_snapshot = _snapshot_model(model)
        optimizer_snapshot = (
            _snapshot_optimizer(optimizer)
            if mode == "resume" and optimizer is not None
            else None
        )
        global_rng_snapshot = torch.get_rng_state().clone()
    except Exception as exc:
        raise CheckpointError(
            "cannot snapshot caller state before checkpoint commit: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    try:
        if mode == "resume" and optimizer is not None:
            assert saved_optimizer_state is not None
            optimizer.load_state_dict(copy.deepcopy(saved_optimizer_state))
        model.load_state_dict(payload["model_state"], strict=True)
        if mode == "resume":
            torch.set_rng_state(validated_rng_state)
    except Exception as exc:
        rollback_errors = []
        try:
            _restore_model(model, model_snapshot)
        except Exception as rollback_exc:  # pragma: no cover - defensive only.
            rollback_errors.append(
                f"model={type(rollback_exc).__name__}: {rollback_exc}"
            )
        if optimizer_snapshot is not None:
            try:
                _restore_optimizer(optimizer, optimizer_snapshot)
            except Exception as rollback_exc:  # pragma: no cover - defensive only.
                rollback_errors.append(
                    f"optimizer={type(rollback_exc).__name__}: {rollback_exc}"
                )
        try:
            torch.set_rng_state(global_rng_snapshot)
        except Exception as rollback_exc:  # pragma: no cover - defensive only.
            rollback_errors.append(
                f"rng={type(rollback_exc).__name__}: {rollback_exc}"
            )
        rollback_suffix = (
            "; rollback errors: " + "; ".join(rollback_errors)
            if rollback_errors
            else ""
        )
        raise CheckpointError(
            "checkpoint commit failed and caller state was rolled back: "
            f"{type(exc).__name__}: {exc}{rollback_suffix}"
        ) from exc
    return CheckpointLoadResult(state, progress, training_state, artifact)


__all__ = [
    "CheckpointArtifact",
    "CheckpointError",
    "CheckpointLoadResult",
    "SCHEMA_VERSION",
    "deserialize_state_store",
    "load_checkpoint",
    "save_checkpoint",
    "serialize_state_store",
]
