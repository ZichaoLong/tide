"""Portable, self-validating SettleGraph equivalence fixture bundles.

The on-disk representation contains only weights-only-safe Python values and
CPU tensors.  Plan bytes are authenticated and checked for canonical form
before they are decoded into :class:`Plan` objects.  The bundle content digest
is independent of the container bytes; :class:`FixtureArtifact.sha256` is the
separate digest of the published file.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import io
import json
import math
import os
import shlex
import stat
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple, Union

import torch
from torch import Tensor

from .checkpoint import (
    CheckpointError,
    deserialize_state_store,
    serialize_state_store,
)
from .engine import StateStore
from .failures import (
    FailureEnvelope,
    FailureEnvelopeError,
    compare_failure_envelopes,
)
from .ops import AttentionState
from .parameter_manifest import (
    PARAMETER_SCHEMA_CANONICALIZER_ID,
    PARAMETER_SCHEMA_VERSION,
    LogicalParameterKey,
    ParameterManifestEntry,
    ParameterManifestError,
    ParameterSchemaManifest,
    build_parameter_schema_manifest,
    export_eager_parameter_tensors,
    logical_parameter_tensor_key,
)
from .plan import (
    PLAN_CANONICALIZER_ID,
    ConcreteBinding,
    EdgeSpec,
    NodeSpec,
    Plan,
    PlanValidationError,
    RegionSpec,
    TypedPlan,
    validate_stable_id,
)


FIXTURE_SCHEMA_VERSION = "tide.settlegraph.fixture.v1"
TENSOR_MANIFEST_SCHEMA_VERSION = "tide.tensor-manifest.v1"


def _is_portable_absolute_path(value: str) -> bool:
    """Recognize absolute POSIX, drive-rooted, and UNC path spellings."""

    windows_path = PureWindowsPath(value)
    return (
        PurePosixPath(value).is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.root)
    )


def _fsync_directory(path: Path) -> None:
    """Commit directory changes while treating descriptor close as cleanup."""

    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, directory_flags)
    try:
        os.fsync(descriptor)
    finally:
        try:
            os.close(descriptor)
        except BaseException:
            # Close diagnostics must not replace an fsync failure.  Once fsync
            # succeeds, close cannot weaken the persisted directory contents.
            pass


def _link_open_file_no_replace(descriptor: int, destination: Path) -> None:
    """Publish the inode behind ``descriptor`` without reopening its name.

    Qualification runs target Linux CPU/CUDA/NPU hosts.  Passing a destination
    directory descriptor makes CPython use ``linkat`` with symlink following;
    Linux then resolves the ``/proc/self/fd`` magic link to the already-open
    staging inode.  A concurrent swap of the staging pathname therefore cannot
    select the bytes published at ``destination``.
    """

    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_handle = os.open(destination.parent, directory_flags)
    try:
        os.link(
            f"/proc/self/fd/{descriptor}",
            destination.name,
            dst_dir_fd=directory_handle,
            follow_symlinks=True,
        )
    finally:
        try:
            os.close(directory_handle)
        except BaseException:
            pass


@contextlib.contextmanager
def _closing_best_effort(stream: Any) -> Iterator[Any]:
    """Close an explicitly fsynced stream without masking a body failure."""

    try:
        yield stream
    finally:
        try:
            stream.close()
        except BaseException:
            pass

_ROUTING_CLASSES = frozenset(
    {"exact-tie", "margin-safe", "near-boundary", "all-active"}
)
_ROOT_KEYS = frozenset(
    {
        "schema_version",
        "fixture_id",
        "source",
        "canonicalizer_id",
        "logical_plan_bytes",
        "logical_plan_hash",
        "typed_plan_bytes",
        "typed_plan_hash",
        "parameter_schema",
        "inputs",
        "parameters",
        "learnable_initial_state",
        "initial_state",
        "control",
        "expected",
        "gradient",
        "routing_classification",
        "tensor_manifest",
        "tensor_artifact_hash",
        "content_hash",
    }
)
_INPUT_KEYS = frozenset(
    {
        "hidden",
        "sequence_ids",
        "token_positions",
        "execution_mask",
        "lm_target_mask",
        "routing_stats_mask",
    }
)
_CONTROL_KEYS = frozenset(
    {
        "requested_k",
        "reset_sequence_ids",
        "chunk_boundaries",
        "detach_boundaries",
        "random_keys",
    }
)
_LOGICAL_PLAN_KEYS = frozenset(
    {
        "schema_version",
        "topology_kind",
        "d_model",
        "dtype_roles",
        "entry_node_ids",
        "terminal_node_ids",
        "output_aggregate",
        "nodes",
        "edges",
        "regions",
    }
)
_NODE_KEYS = frozenset(
    {
        "node_id",
        "region_id",
        "hidden_shape",
        "selector_read_shape",
        "state_shape",
        "state_owner",
        "forced_active",
        "parameter_group",
        "input_norm",
        "ffn_norm",
        "aggregate",
        "update",
        "selector_read",
        "ffn_read",
        "node_compute",
        "emit",
    }
)
_EDGE_KEYS = frozenset({"edge_id", "source", "target", "label"})
_REGION_KEYS = frozenset(
    {
        "region_id",
        "node_ids",
        "profile",
        "selector_timing",
        "k_max",
        "k_requested",
        "score",
        "selector_context",
        "selector_history",
        "control_dependencies",
        "line",
        "phase",
    }
)
_TYPED_PLAN_KEYS = frozenset(
    {"schema_version", "logical_plan_hash", "binding"}
)
_GRADIENT_KEYS = frozenset(
    {
        "objective",
        "output_cotangent",
        "alpha_lm",
        "alpha_balance",
        "required_keys",
        "path_assertions",
    }
)
_PATH_ASSERTIONS = frozenset({"connected", "disconnected", "absent"})
_NEGATIVE_MUTATIONS = frozenset(
    {
        "plan.topology.repeat-region-member",
        "input.mask.lm-outside-execution",
        "state.owner-alias.nonoverlap-view",
    }
)


class FixtureError(ValueError):
    """A fixture fails one stable artifact, Plan, binding, input, or state gate."""

    def __init__(
        self,
        phase: str,
        codes: Union[str, Iterable[str]],
        message: str,
    ) -> None:
        self.envelope = FailureEnvelope.create(phase, codes)
        self.phase = self.envelope.phase
        self.codes = self.envelope.codes
        # Kept for callers written before multi-code Plan diagnostics existed.
        self.code = self.codes[0]
        super().__init__(message)


@dataclasses.dataclass(frozen=True)
class FixtureArtifact:
    """Identity of one published fixture container."""

    path: str
    sha256: str
    content_hash: str
    tensor_artifact_hash: str


@dataclasses.dataclass(frozen=True)
class FixtureBundle:
    """A validated, CPU-resident equivalence fixture."""

    fixture_id: str
    typed_plan: TypedPlan
    source: Mapping[str, Any]
    parameter_schema: Mapping[str, Any]
    inputs: Mapping[str, Any]
    parameters: Mapping[str, Tensor]
    learnable_initial_state: Mapping[str, Tensor]
    initial_state: StateStore
    control: Mapping[str, Any]
    expected: Mapping[str, Any]
    gradient: Mapping[str, Any]
    routing_classification: str
    tensor_manifest: Tuple[Mapping[str, Any], ...]
    artifact: FixtureArtifact


def _raise(
    phase: str, codes: Union[str, Iterable[str]], message: str
) -> None:
    raise FixtureError(phase, codes, message)


def _tensor_bytes(value: Tensor) -> bytes:
    contiguous = value.detach().to(device="cpu").contiguous().reshape(-1)
    return contiguous.view(torch.uint8).numpy().tobytes()


def _tensor_sha256(value: Tensor) -> str:
    return hashlib.sha256(_tensor_bytes(value)).hexdigest()


def _storage_sha256(value: Tensor) -> str:
    """Hash every byte in a CPU Tensor's backing storage, including holes."""

    storage = value.untyped_storage()
    byte_count = int(storage.nbytes())
    byte_view = torch.empty(0, dtype=torch.uint8, device="cpu")
    byte_view.set_(storage, 0, (byte_count,), (1,))
    return hashlib.sha256(byte_view.numpy().tobytes()).hexdigest()


def _certified_layout_span(value: Tensor) -> Optional[int]:
    """Return the required element span for a supported non-overlap layout.

    Sorting non-singleton dimensions by stride gives a constructive
    certificate: every new stride must begin at or beyond the complete span
    of the lower-stride dimensions.  This admits dense permutations and
    positive-stride slices with holes, while conservatively rejecting exotic
    layouts whose lack of overlap cannot be established by this v1 loader.
    """

    shape = tuple(value.shape)
    strides = tuple(value.stride())
    if len(shape) != len(strides) or any(stride < 0 for stride in strides):
        return None
    if value.numel() == 0:
        return 0
    span = 1
    for stride, size in sorted(
        (stride, size)
        for size, stride in zip(shape, strides)
        if size > 1
    ):
        if stride <= 0 or stride < span:
            return None
        span += (size - 1) * stride
    return span


def _cpu_tensor(value: Tensor) -> Tensor:
    if value.layout != torch.strided:
        _raise(
            "artifact",
            "artifact.schema",
            "fixture tensors must use the strided layout",
        )
    span = _certified_layout_span(value)
    storage_offset = int(value.storage_offset())
    if span is None or storage_offset < 0:
        _raise(
            "artifact",
            "artifact.schema",
            "fixture tensors require a certified non-overlapping "
            "nonnegative-stride layout",
        )
    try:
        # Zero the holes so serialization never persists unrelated backing
        # storage.  The artifact preserves shape, stride and storage offset,
        # using only the minimal required span.
        backing = torch.zeros(
            storage_offset + span,
            dtype=value.dtype,
            device="cpu",
        )
        copied = torch.as_strided(
            backing,
            size=tuple(value.shape),
            stride=tuple(value.stride()),
            storage_offset=storage_offset,
        )
        copied.copy_(value.detach().to(device="cpu"))
        return copied
    except (NotImplementedError, RuntimeError, TypeError, ValueError) as exc:
        _raise(
            "artifact",
            "artifact.schema",
            "fixture Tensor layout/dtype cannot be copied safely: "
            f"{type(exc).__name__}: {exc}",
        )


def _safe_value(value: Any, *, path: str) -> Any:
    """Copy one value into the weights-only-safe fixture subset."""

    if isinstance(value, str):
        if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
            _raise(
                "artifact",
                "artifact.schema",
                f"{path} must contain only Unicode scalar values",
            )
        return value
    if value is None or isinstance(value, (bool, int, bytes)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            _raise(
                "artifact",
                "artifact.schema",
                f"{path} must not contain NaN or infinity metadata",
            )
        return value
    if isinstance(value, Tensor):
        return _cpu_tensor(value)
    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                _raise(
                    "artifact",
                    "artifact.schema",
                    f"{path} mapping keys must be strings",
                )
            if any(0xD800 <= ord(character) <= 0xDFFF for character in key):
                _raise(
                    "artifact",
                    "artifact.schema",
                    f"{path} mapping keys must contain only Unicode scalar values",
                )
            if (
                key.endswith("_path")
                and isinstance(item, str)
                and _is_portable_absolute_path(item)
            ):
                _raise(
                    "artifact",
                    "artifact.schema",
                    f"{path}.{key} must not be an absolute private path",
                )
            result[key] = _safe_value(item, path=f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _safe_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    _raise(
        "artifact",
        "artifact.schema",
        f"{path} contains unsupported {type(value).__name__}",
    )


def _json_pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _walk_tensor_pairs(value: Any, *, path: str = "") -> Iterable[Tuple[str, Tensor]]:
    if isinstance(value, Tensor):
        yield path or "/", value
    elif isinstance(value, Mapping):
        for key in sorted(value):
            if not isinstance(key, str):
                _raise(
                    "artifact",
                    "artifact.schema",
                    "fixture mapping keys must be strings",
                )
            yield from _walk_tensor_pairs(
                value[key], path=path + "/" + _json_pointer_token(key)
            )
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _walk_tensor_pairs(item, path=path + "/" + str(index))


def _tensor_role(path: str) -> str:
    if path.startswith("/inputs/hidden"):
        return "hidden"
    if path.startswith("/parameters/") or path.startswith(
        "/learnable_initial_state/"
    ):
        return "parameter"
    if path.startswith("/initial_state/"):
        return "state"
    if path.startswith("/inputs/") or path.startswith("/control/"):
        return "control"
    if path.startswith("/gradient/"):
        return "gradient"
    if path.startswith("/expected/"):
        return "expected"
    return "artifact"


def _build_tensor_manifest(payload: Mapping[str, Any]) -> list[Dict[str, Any]]:
    pairs = list(_walk_tensor_pairs(payload))
    parents = list(range(len(pairs)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    storage_records = []
    for _, value in pairs:
        storage = value.untyped_storage()
        start = storage.data_ptr()
        size = storage.nbytes()
        storage_records.append(
            (
                value.device.type,
                value.device.index,
                getattr(storage, "_cdata", None),
                start,
                start + size,
                size,
            )
        )
    for left in range(len(storage_records)):
        left_type, left_index, left_id, left_start, left_end, left_size = (
            storage_records[left]
        )
        if left_size <= 0:
            continue
        for right in range(left):
            right_type, right_index, right_id, right_start, right_end, right_size = (
                storage_records[right]
            )
            if right_size <= 0 or (left_type, left_index) != (
                right_type,
                right_index,
            ):
                continue
            same_storage = left_id is not None and left_id == right_id
            overlapping_storage = (
                left_start
                and right_start
                and left_start < right_end
                and right_start < left_end
            )
            if same_storage or overlapping_storage:
                union(left, right)
    roots = sorted(
        {find(index) for index in range(len(pairs))},
        key=lambda root: min(
            pairs[index][0]
            for index in range(len(pairs))
            if find(index) == root
        ),
    )
    group_by_root = {
        root: f"storage.{ordinal:08d}" for ordinal, root in enumerate(roots)
    }

    entries = []
    for index, (path, value) in enumerate(pairs):
        if value.device.type != "cpu":
            _raise(
                "artifact",
                "artifact.schema",
                f"fixture tensor {path} must be stored on CPU",
            )
        entries.append(
            {
                "path": path,
                "role": _tensor_role(path),
                "shape": list(value.shape),
                "stride": list(value.stride()),
                "dtype": str(value.dtype).removeprefix("torch."),
                "nbytes": len(_tensor_bytes(value)),
                "storage_offset": value.storage_offset(),
                "storage_nbytes": value.untyped_storage().nbytes(),
                "storage_group": group_by_root[find(index)],
                "storage_sha256": _storage_sha256(value),
                "sha256": _tensor_sha256(value),
            }
        )
    return entries


def _canonical_digest_value(value: Any) -> Any:
    """Encode every supported value kind in a disjoint digest domain."""

    if isinstance(value, Tensor):
        return [
            "tensor",
            {
                "shape": list(value.shape),
                "dtype": str(value.dtype).removeprefix("torch."),
                "sha256": _tensor_sha256(value),
            },
        ]
    if isinstance(value, bytes):
        return ["bytes", value.hex()]
    if isinstance(value, Mapping):
        return [
            "mapping",
            [
                [key, _canonical_digest_value(value[key])]
                for key in sorted(value)
            ],
        ]
    if isinstance(value, list):
        return ["list", [_canonical_digest_value(item) for item in value]]
    if isinstance(value, tuple):
        return ["tuple", [_canonical_digest_value(item) for item in value]]
    if value is None:
        return ["none"]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", value]
    if isinstance(value, str):
        return ["string", value]
    if isinstance(value, float) and math.isfinite(value):
        return ["float", value]
    _raise(
        "artifact",
        "artifact.schema",
        f"fixture digest encountered unsupported {type(value).__name__}",
    )


def _canonical_digest_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            _canonical_digest_value(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        _raise(
            "artifact",
            "artifact.schema",
            "fixture digest is not canonical UTF-8 JSON: "
            f"{type(exc).__name__}: {exc}",
        )


def _canonical_json_bytes(value: Any) -> bytes:
    """Render the ordinary canonical JSON used by logical/typed Plan bytes."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        _raise(
            "artifact",
            "artifact.schema",
            "record is not canonical UTF-8 JSON: "
            f"{type(exc).__name__}: {exc}",
        )


def _record_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_digest_bytes(value)).hexdigest()


def _exact_mapping(
    value: Any, keys: frozenset[str], *, context: str, phase: str, code: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        _raise(
            phase,
            code,
            f"{context} must be a mapping with exactly {sorted(keys)!r}",
        )
    return value


class _DuplicateJsonKey(ValueError):
    pass


def _reject_duplicate_json_keys(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey(key)
        result[key] = value
    return result


def _decode_json_bytes(data: Any, *, context: str) -> Any:
    if not isinstance(data, bytes):
        _raise("artifact", "artifact.schema", f"{context} must be bytes")
    try:
        text = data.decode("utf-8", errors="strict")
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        _raise(
            "artifact",
            "artifact.integrity",
            f"{context} is not valid canonical UTF-8 JSON: {exc}",
        )


def _decode_logical_plan(record: Any, *, plan_id: str) -> Plan:
    root = _exact_mapping(
        record,
        _LOGICAL_PLAN_KEYS,
        context="logical Plan",
        phase="plan",
        code="plan.schema",
    )
    nodes_raw = root["nodes"]
    edges_raw = root["edges"]
    regions_raw = root["regions"]
    if not isinstance(nodes_raw, list) or not isinstance(edges_raw, list) or not isinstance(
        regions_raw, list
    ):
        _raise(
            "plan",
            "plan.schema",
            "logical Plan nodes, edges, and regions must be JSON arrays",
        )
    try:
        nodes = []
        for index, raw in enumerate(nodes_raw):
            item = _exact_mapping(
                raw,
                _NODE_KEYS,
                context=f"logical Plan node {index}",
                phase="plan",
                code="plan.schema",
            )
            nodes.append(NodeSpec(**dict(item)))
        edges = []
        for index, raw in enumerate(edges_raw):
            item = _exact_mapping(
                raw,
                _EDGE_KEYS,
                context=f"logical Plan edge {index}",
                phase="plan",
                code="plan.schema",
            )
            edges.append(EdgeSpec(**dict(item)))
        regions = []
        for index, raw in enumerate(regions_raw):
            item = _exact_mapping(
                raw,
                _REGION_KEYS,
                context=f"logical Plan region {index}",
                phase="plan",
                code="plan.schema",
            )
            regions.append(RegionSpec(**dict(item)))
        plan = Plan(
            plan_id=plan_id,
            d_model=root["d_model"],
            dtype_roles=root["dtype_roles"],
            nodes=tuple(nodes),
            edges=tuple(edges),
            regions=tuple(regions),
            # Plan owns sequence normalization.  Passing raw values through is
            # essential: a JSON string is not an array and must not be silently
            # split into one-character stable IDs by tuple(...).
            entry_node_ids=root["entry_node_ids"],
            terminal_node_ids=root["terminal_node_ids"],
            output_aggregate=root["output_aggregate"],
            topology_kind=root["topology_kind"],
            schema_version=root["schema_version"],
        )
        return plan.validate()
    except FixtureError:
        raise
    except PlanValidationError as exc:
        codes = getattr(exc, "failure_codes", ()) or ("plan.schema",)
        _raise("plan", codes, f"logical Plan is invalid: {exc}")
    except (AttributeError, TypeError, ValueError, KeyError) as exc:
        _raise("plan", "plan.schema", f"logical Plan cannot be decoded: {exc}")


def _decode_typed_plan(
    logical_plan: Plan,
    record: Any,
    *,
    expected_logical_hash: str,
) -> TypedPlan:
    typed = _exact_mapping(
        record,
        _TYPED_PLAN_KEYS,
        context="typed Plan",
        phase="binding",
        code="binding.invalid",
    )
    binding = typed["binding"]
    if (
        not isinstance(binding, Mapping)
        or set(binding) != {"dtype_roles"}
        or not isinstance(binding["dtype_roles"], Mapping)
    ):
        _raise(
            "binding",
            "binding.invalid",
            "typed Plan binding must contain exactly dtype_roles",
        )
    if typed["schema_version"] != "1" or typed["logical_plan_hash"] != expected_logical_hash:
        _raise(
            "binding",
            "binding.invalid",
            "typed Plan schema or logical Plan identity is invalid",
        )
    try:
        return TypedPlan(
            logical_plan,
            ConcreteBinding(dict(binding["dtype_roles"])),
        ).validate()
    except (PlanValidationError, TypeError, ValueError) as exc:
        _raise("binding", "binding.invalid", f"typed Plan is invalid: {exc}")


def _as_string_tensor_mapping(value: Any, *, context: str) -> Mapping[str, Tensor]:
    if not isinstance(value, Mapping):
        _raise("artifact", "artifact.schema", f"{context} must be a mapping")
    result: Dict[str, Tensor] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            _raise(
                "artifact",
                "artifact.schema",
                f"{context} keys must be nonempty strings",
            )
        if not isinstance(item, Tensor):
            _raise(
                "artifact",
                "artifact.schema",
                f"{context}[{key!r}] must be a Tensor",
            )
        if item.device.type != "cpu" or item.layout != torch.strided:
            _raise(
                "artifact",
                "artifact.schema",
                f"{context}[{key!r}] must be a CPU strided Tensor",
            )
        result[key] = item
    return MappingProxyType(result)


def _dtype_from_binding(typed_plan: TypedPlan, role: str) -> torch.dtype:
    name = typed_plan.binding.dtype_roles[role]
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        _raise(
            "binding",
            "binding.invalid",
            f"unknown concrete dtype {name!r} for role {role!r}",
        )
    return dtype


def _validate_expected(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _raise("artifact", "artifact.schema", "expected must be a mapping")
    outcome = value.get("outcome")
    if not isinstance(outcome, str) or outcome not in {"success", "failure"}:
        _raise(
            "artifact",
            "artifact.schema",
            "expected.outcome must be 'success' or 'failure'",
        )
    if outcome == "failure":
        if set(value) != {"outcome", "error"}:
            _raise(
                "artifact",
                "artifact.schema",
                "failure expectation must contain exactly outcome and error",
            )
        try:
            FailureEnvelope.from_dict(value["error"])
        except FailureEnvelopeError as exc:
            _raise(
                "artifact",
                "artifact.schema",
                f"expected failure envelope is invalid: {exc}",
            )
    elif "error" in value:
        _raise(
            "artifact",
            "artifact.schema",
            "a success expectation must not contain an error envelope",
        )
    return value


def _validate_source(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _raise("artifact", "artifact.schema", "source must be a mapping")
    for required in ("kind", "identifier", "command"):
        item = value.get(required)
        if not isinstance(item, str) or not item:
            _raise(
                "artifact",
                "artifact.schema",
                f"source.{required} must be a nonempty string",
            )
    if _is_portable_absolute_path(value["identifier"]):
        _raise(
            "artifact",
            "artifact.schema",
            "source.identifier must not be an absolute path",
        )
    try:
        command_tokens = shlex.split(value["command"])
    except ValueError as exc:
        _raise(
            "artifact",
            "artifact.schema",
            f"source.command is not valid shell-like text: {exc}",
        )
    # The bundle remains portable when produced on POSIX, so also inspect a
    # tokenization that preserves Windows backslashes.  The POSIX parse above
    # remains the authority for command syntax.
    portable_tokens = list(command_tokens)
    try:
        portable_tokens.extend(shlex.split(value["command"], posix=False))
    except ValueError:
        pass
    for token in portable_tokens:
        candidate = token.split("=", 1)[1] if "=" in token else token
        candidate = candidate.strip("'\"")
        # shlex.split keeps attached shell redirections such as ``2>/path``
        # in one token.  Inspect the redirected path rather than the operator.
        if ">" in candidate or "<" in candidate:
            candidate = candidate[max(candidate.rfind(">"), candidate.rfind("<")) + 1 :]
            candidate = candidate.strip("'\"")
        if _is_portable_absolute_path(candidate):
            _raise(
                "artifact",
                "artifact.schema",
                "source.command must not contain an absolute path",
            )
    mutation = value.get("mutation_kind")
    allowed = {"kind", "identifier", "command", "mutation_kind"}
    if set(value) - allowed:
        _raise(
            "artifact",
            "artifact.schema",
            "source contains unsupported provenance keys",
        )
    if mutation is not None and (
        not isinstance(mutation, str) or mutation not in _NEGATIVE_MUTATIONS
    ):
        _raise(
            "artifact",
            "artifact.schema",
            "source.mutation_kind is not a supported named mutation",
        )
    return value


def _validate_gradient(
    value: Any,
    *,
    hidden: Tensor,
    parameters: Mapping[str, Tensor],
    learnable_initial_state: Mapping[str, Tensor],
    standalone: bool = True,
) -> Mapping[str, Any]:
    gradient = _exact_mapping(
        value,
        _GRADIENT_KEYS,
        context="gradient",
        phase="artifact",
        code="artifact.schema",
    )
    if gradient["objective"] != "output-cotangent-v1":
        _raise(
            "artifact",
            "artifact.schema",
            "gradient.objective must be 'output-cotangent-v1'",
        )
    cotangent = gradient["output_cotangent"]
    if (
        not isinstance(cotangent, Tensor)
        or cotangent.device.type != "cpu"
        or cotangent.layout != torch.strided
        or cotangent.shape != hidden.shape
        or cotangent.dtype != hidden.dtype
    ):
        _raise(
            "artifact",
            "artifact.schema",
            "gradient.output_cotangent must match the CPU hidden shape and dtype",
        )
    for name in ("alpha_lm", "alpha_balance"):
        coefficient = gradient[name]
        if (
            isinstance(coefficient, bool)
            or not isinstance(coefficient, (int, float))
            or not math.isfinite(float(coefficient))
        ):
            _raise(
                "artifact",
                "artifact.schema",
                f"gradient.{name} must be a finite number",
            )
    if standalone and float(gradient["alpha_lm"]) != 0.0:
        _raise(
            "artifact",
            "artifact.schema",
            "standalone SettleGraph fixtures require gradient.alpha_lm=0",
        )
    keys = gradient["required_keys"]
    assertions = gradient["path_assertions"]
    required_paths = {
        "inputs.hidden",
        *(f"parameters.{key}" for key in parameters),
        *(
            f"learnable_initial_state.{key}"
            for key in learnable_initial_state
        ),
    }
    if (
        not isinstance(keys, (list, tuple))
        or any(not isinstance(key, str) or not key for key in keys)
        or list(keys) != sorted(set(keys))
        or set(keys) != required_paths
        or not isinstance(assertions, Mapping)
        or set(assertions) != set(keys)
        or any(
            not isinstance(assertion, str)
            or assertion not in {"connected", "disconnected"}
            for assertion in assertions.values()
        )
    ):
        _raise(
            "artifact",
            "artifact.schema",
            "gradient required_keys/path_assertions must be the sorted, exact "
            "contract for hidden, parameters, and learnable initial state",
        )
    return gradient


def _validate_distinct_tensor_storage(
    mappings: Iterable[Tuple[str, Mapping[str, Tensor]]],
) -> None:
    owners: Dict[Tuple[str, Optional[int], int], str] = {}
    ranges: list[Tuple[str, Optional[int], int, int, str]] = []
    for namespace, values in mappings:
        for key, tensor in values.items():
            owner = f"{namespace}[{key!r}]"
            storage = tensor.untyped_storage()
            identity = (tensor.device.type, tensor.device.index, storage._cdata)
            other = owners.get(identity)
            if other is not None and other != owner:
                _raise(
                    "artifact",
                    "artifact.schema",
                    f"serialized parameter storage is shared by {other} and {owner}",
                )
            start = storage.data_ptr()
            end = start + storage.nbytes()
            if start and end > start:
                for device_type, device_index, other_start, other_end, other in ranges:
                    if (
                        device_type == tensor.device.type
                        and device_index == tensor.device.index
                        and start < other_end
                        and other_start < end
                        and other != owner
                    ):
                        _raise(
                            "artifact",
                            "artifact.schema",
                            f"serialized parameter storage is shared by {other} and {owner}",
                        )
                ranges.append(
                    (tensor.device.type, tensor.device.index, start, end, owner)
                )
            owners[identity] = owner


def _validate_parameter_schema(
    value: Any, logical_hash: str, plan: Optional[Plan] = None
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _raise(
            "artifact", "artifact.schema", "parameter_schema must be a mapping"
        )
    if not value:
        _raise(
            "artifact",
            "artifact.schema",
            "parameter_schema must contain the Plan-derived canonical manifest",
        )
    if value:
        if set(value) != {
            "schema_version",
            "canonicalizer_id",
            "logical_plan_hash",
            "parameters",
        }:
            _raise(
                "artifact",
                "artifact.schema",
                "parameter_schema root has an unexpected key set",
            )
        if value.get("schema_version") != PARAMETER_SCHEMA_VERSION:
            _raise(
                "artifact",
                "artifact.schema",
                "parameter_schema has an unsupported schema_version",
            )
        if value.get("canonicalizer_id") != PARAMETER_SCHEMA_CANONICALIZER_ID:
            _raise(
                "artifact",
                "artifact.schema",
                "parameter_schema has an unsupported canonicalizer_id",
            )
        if value.get("logical_plan_hash") != logical_hash:
            _raise(
                "artifact",
                "artifact.schema",
                "parameter_schema logical Plan hash does not match",
            )
        entries = value.get("parameters")
        if not isinstance(entries, (list, tuple)):
            _raise(
                "artifact",
                "artifact.schema",
                "parameter_schema.parameters must be an array",
            )
        decoded_entries = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping) or set(entry) != {
                "logical_key",
                "formula_id",
                "shape",
                "dtype_role",
                "parameter_group",
            }:
                _raise(
                    "artifact",
                    "artifact.schema",
                    f"parameter_schema.parameters[{index}] is malformed",
                )
            logical_key = entry["logical_key"]
            if not isinstance(logical_key, Mapping) or set(logical_key) != {
                "field",
                "region_id",
                "node_id",
                "edge_id",
                "terminal_node_id",
                "parameter_role",
            }:
                _raise(
                    "artifact",
                    "artifact.schema",
                    f"parameter_schema.parameters[{index}].logical_key is malformed",
                )
            try:
                decoded_entries.append(
                    ParameterManifestEntry(
                        logical_key=LogicalParameterKey(**dict(logical_key)),
                        formula_id=entry["formula_id"],
                        shape=tuple(entry["shape"]),
                        dtype_role=entry["dtype_role"],
                        parameter_group=entry["parameter_group"],
                        state_dict_locator=f"fixture.parameter.{index:08d}",
                    )
                )
            except (ParameterManifestError, TypeError, ValueError) as exc:
                _raise(
                    "artifact",
                    "artifact.schema",
                    f"parameter_schema.parameters[{index}] metadata is invalid: {exc}",
                )
        try:
            decoded = ParameterSchemaManifest(logical_hash, tuple(decoded_entries))
        except ParameterManifestError as exc:
            _raise(
                "artifact",
                "artifact.schema",
                f"parameter_schema is invalid: {exc}",
            )
        if decoded.canonical_dict() != dict(value):
            _raise(
                "artifact",
                "artifact.schema",
                "parameter_schema is not in canonical entry order",
            )
        if plan is not None:
            try:
                expected = build_parameter_schema_manifest(plan).canonical_dict()
            except ParameterManifestError as exc:
                _raise(
                    "plan",
                    "plan.formula",
                    f"cannot derive the Plan parameter schema: {exc}",
                )
            if dict(value) != expected:
                _raise(
                    "artifact",
                    "artifact.schema",
                    "parameter_schema does not match the Plan formula parameters",
                )
    return value


def eager_parameter_tensors(
    model: torch.nn.Module, manifest: ParameterSchemaManifest
) -> Mapping[str, Tensor]:
    """Export eager parameters under implementation-independent logical keys."""

    return MappingProxyType(dict(export_eager_parameter_tensors(model, manifest)))


def _validate_state_for_plan(store: StateStore, typed_plan: TypedPlan) -> None:
    plan = typed_plan.logical_plan
    state_dtype = _dtype_from_binding(typed_plan, "state")
    known = {node.node_id: node for node in plan.nodes}
    storage_owner: Dict[Tuple[str, Optional[int], int], Tuple[str, str]] = {}
    storage_ranges: list[
        Tuple[str, Optional[int], int, int, Tuple[str, str]]
    ] = []
    if store.selector_history:
        _raise(
            "state",
            "state.schema",
            "current standard fixture subset requires empty selector_history",
        )
    for (sequence_id, node_id), state in store.values.items():
        node = known.get(node_id)
        if node is None:
            _raise("state", "state.schema", f"state names unknown node {node_id!r}")
        if sequence_id not in store.next_position:
            _raise(
                "state",
                "state.schema",
                f"state sequence {sequence_id!r} has no next position",
            )
        update_type = str(node.update.get("type", "none")).lower().replace("-", "_")
        tensors: Tuple[Tensor, ...]
        floating: Tuple[Tensor, ...]
        if update_type == "none":
            _raise(
                "state",
                "state.schema",
                f"stateless node {node_id!r} must not have stored state",
            )
        if update_type == "attention_window":
            if not isinstance(state, AttentionState):
                _raise(
                    "state",
                    "state.schema",
                    f"node {node_id!r} requires canonical AttentionState",
                )
            length = state.length
            key_dim = node.update.get("key_dim")
            value_dim = node.update.get("value_dim")
            window = node.update.get("window")
            if (
                state.positions.dtype != torch.int64
                or state.positions.shape != (length,)
                or state.keys.shape != (length, key_dim)
                or state.values.shape != (length, value_dim)
                or type(window) is not int
                or length > window
            ):
                _raise(
                    "state",
                    "state.schema",
                    f"Attention state for {node_id!r} has invalid shape or window",
                )
            if length > 1 and bool(
                (state.positions[1:] <= state.positions[:-1]).any().item()
            ):
                _raise(
                    "state",
                    "state.schema",
                    "Attention positions must increase strictly",
                )
            if length and (
                int(state.positions[0].item()) < 0
                or int(state.positions[-1].item()) >= store.next_position[sequence_id]
            ):
                _raise(
                    "state",
                    "state.schema",
                    "Attention positions must precede the sequence next position",
                )
            tensors = (state.positions, state.keys, state.values)
            floating = (state.keys, state.values)
        else:
            if not isinstance(state, Tensor) or tuple(state.shape) != node.state_shape:
                _raise(
                    "state",
                    "state.schema",
                    f"state for {node_id!r} does not match declared shape {node.state_shape!r}",
                )
            tensors = (state,)
            floating = (state,)
        for tensor in tensors:
            if tensor.device.type != "cpu":
                _raise("state", "state.schema", "fixture state must reside on CPU")
            storage = tensor.untyped_storage()
            identity = (tensor.device.type, tensor.device.index, storage._cdata)
            key = (sequence_id, node_id)
            other = storage_owner.get(identity)
            if other is not None and other != key:
                _raise(
                    "state",
                    "state.owner_alias",
                    f"mutable state storage is shared by {other!r} and {key!r}",
                )
            start = storage.data_ptr()
            end = start + storage.nbytes()
            if start and end > start:
                for device_type, device_index, other_start, other_end, other_key in storage_ranges:
                    if (
                        other_key != key
                        and device_type == tensor.device.type
                        and device_index == tensor.device.index
                        and start < other_end
                        and other_start < end
                    ):
                        _raise(
                            "state",
                            "state.owner_alias",
                            f"mutable state storage is shared by {other_key!r} and {key!r}",
                        )
                storage_ranges.append(
                    (tensor.device.type, tensor.device.index, start, end, key)
                )
            storage_owner[identity] = key
        if any(
            not tensor.is_floating_point() or tensor.dtype != state_dtype
            for tensor in floating
        ):
            _raise(
                "state",
                "state.schema",
                f"floating state for {node_id!r} must use {state_dtype}",
            )


def _validate_inputs_and_control(
    inputs: Any,
    control: Any,
    typed_plan: TypedPlan,
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    inputs = _exact_mapping(
        inputs,
        _INPUT_KEYS,
        context="inputs",
        phase="artifact",
        code="artifact.schema",
    )
    control = _exact_mapping(
        control,
        _CONTROL_KEYS,
        context="control",
        phase="artifact",
        code="artifact.schema",
    )
    hidden = inputs["hidden"]
    execution = inputs["execution_mask"]
    positions = inputs["token_positions"]
    lm_mask = inputs["lm_target_mask"]
    routing_mask = inputs["routing_stats_mask"]
    if (
        not isinstance(hidden, Tensor)
        or hidden.device.type != "cpu"
        or hidden.ndim != 3
        or hidden.shape[2] != typed_plan.logical_plan.d_model
        or hidden.dtype != _dtype_from_binding(typed_plan, "hidden")
    ):
        _raise(
            "input",
            "input.schema",
            "hidden must be a CPU [B,T,d_model] Tensor in the bound dtype",
        )
    batch, length, _ = hidden.shape
    sequence_ids = inputs["sequence_ids"]
    if not isinstance(sequence_ids, (list, tuple)) or len(sequence_ids) != batch:
        _raise(
            "input", "input.schema", "sequence_ids length must equal batch size"
        )
    validated_ids = []
    for value in sequence_ids:
        try:
            validated_ids.append(validate_stable_id(value, kind="sequence"))
        except ValueError as exc:
            _raise("input", "input.schema", str(exc))
    if len(set(validated_ids)) != len(validated_ids) or validated_ids != sorted(
        validated_ids
    ):
        _raise(
            "input",
            "input.schema",
            "fixture sequence_ids must be unique and in stable ascending order",
        )
    if (
        not isinstance(positions, Tensor)
        or positions.device.type != "cpu"
        or positions.dtype != torch.int64
        or positions.shape != (batch, length)
    ):
        _raise(
            "input",
            "input.schema",
            "token_positions must be a CPU int64 Tensor with shape [B,T]",
        )
    for name, mask in (
        ("execution_mask", execution),
        ("lm_target_mask", lm_mask),
        ("routing_stats_mask", routing_mask),
    ):
        if (
            not isinstance(mask, Tensor)
            or mask.device.type != "cpu"
            or mask.dtype != torch.bool
            or mask.shape != (batch, length)
        ):
            _raise(
                "input",
                "input.mask",
                f"{name} must be a CPU bool Tensor with shape [B,T]",
            )
    if bool((lm_mask & ~execution).any().item()) or bool(
        (routing_mask & ~execution).any().item()
    ):
        _raise(
            "input",
            "input.mask",
            "LM and routing-stat masks must be subsets of execution_mask",
        )

    reset_ids = control["reset_sequence_ids"]
    if not isinstance(reset_ids, (list, tuple)):
        _raise(
            "input", "input.schema", "reset_sequence_ids must be an array"
        )
    validated_reset = []
    for value in reset_ids:
        try:
            validated_reset.append(validate_stable_id(value, kind="reset sequence"))
        except ValueError as exc:
            _raise("input", "input.schema", str(exc))
    if validated_reset != sorted(set(validated_reset)):
        _raise(
            "input",
            "input.schema",
            "reset_sequence_ids must be unique and in stable ascending order",
        )

    requested = control["requested_k"]
    if not isinstance(requested, Mapping):
        _raise("input", "input.schema", "requested_k must be a mapping")
    open_regions = {
        region.region_id
        for region in typed_plan.logical_plan.regions
        if region.k_requested.get("type") == "input"
    }
    if set(requested) != open_regions:
        _raise(
            "input",
            "input.schema",
            f"requested_k keys must exactly equal {sorted(open_regions)!r}",
        )
    for region_id, values in requested.items():
        if (
            not isinstance(values, Tensor)
            or values.device.type != "cpu"
            or values.dtype != torch.int64
            or values.shape != (batch, length)
        ):
            _raise(
                "input",
                "input.schema",
                f"requested_k[{region_id!r}] must be CPU int64 [B,T]",
            )

    def validate_boundaries(name: str, values: Any) -> Tuple[int, ...]:
        if not isinstance(values, (list, tuple)) or any(
            type(item) is not int for item in values
        ):
            _raise("input", "input.schema", f"{name} must be an integer array")
        result = tuple(values)
        if result != tuple(sorted(set(result))) or any(
            item <= 0 or item >= length for item in result
        ):
            _raise(
                "input",
                "input.schema",
                f"{name} must contain unique ascending boundaries in [1,T-1]",
            )
        return result

    chunk_boundaries = validate_boundaries(
        "chunk_boundaries", control["chunk_boundaries"]
    )
    detach_boundaries = validate_boundaries(
        "detach_boundaries", control["detach_boundaries"]
    )
    if not set(detach_boundaries).issubset(chunk_boundaries):
        _raise(
            "input",
            "input.schema",
            "detach_boundaries must be a subset of chunk_boundaries",
        )
    if not isinstance(control["random_keys"], Mapping):
        _raise("input", "input.schema", "random_keys must be a mapping")
    return inputs, control


def _validate_positions(
    inputs: Mapping[str, Any],
    control: Mapping[str, Any],
    initial_state: StateStore,
) -> None:
    sequence_ids = tuple(inputs["sequence_ids"])
    reset_ids = frozenset(control["reset_sequence_ids"])
    execution = inputs["execution_mask"]
    positions = inputs["token_positions"]
    expected_positions = {
        sequence_id: (
            0
            if sequence_id in reset_ids
            else initial_state.next_position.get(sequence_id, 0)
        )
        for sequence_id in sequence_ids
    }
    for token_index in range(execution.shape[1]):
        for row, sequence_id in enumerate(sequence_ids):
            if not bool(execution[row, token_index].item()):
                continue
            actual = int(positions[row, token_index].item())
            if actual != expected_positions[sequence_id]:
                _raise(
                    "input",
                    "input.position",
                    f"position {actual} for {sequence_id!r} must equal "
                    f"{expected_positions[sequence_id]}",
                )
            expected_positions[sequence_id] += 1


def _verify_tensor_manifest(payload: Mapping[str, Any]) -> Tuple[Mapping[str, Any], ...]:
    raw = payload["tensor_manifest"]
    if (
        not isinstance(raw, Mapping)
        or set(raw) != {"schema_version", "tensors"}
        or raw["schema_version"] != TENSOR_MANIFEST_SCHEMA_VERSION
        or not isinstance(raw["tensors"], (list, tuple))
    ):
        _raise("artifact", "artifact.schema", "tensor_manifest schema is invalid")
    content = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "tensor_manifest",
            "tensor_artifact_hash",
            "content_hash",
        }
    }
    actual = _build_tensor_manifest(content)
    expected = list(raw["tensors"])
    entry_keys = {
        "path",
        "role",
        "shape",
        "stride",
        "dtype",
        "nbytes",
        "storage_offset",
        "storage_nbytes",
        "storage_group",
        "storage_sha256",
        "sha256",
    }
    allowed_roles = {
        "hidden",
        "parameter",
        "state",
        "control",
        "gradient",
        "expected",
        "artifact",
    }

    def valid_entry(entry: Any) -> bool:
        if not isinstance(entry, Mapping) or set(entry) != entry_keys:
            return False
        shape = entry["shape"]
        stride = entry["stride"]
        digest = entry["sha256"]
        storage_digest = entry["storage_sha256"]
        return (
            isinstance(entry["path"], str)
            and entry["path"].startswith("/")
            and isinstance(entry["role"], str)
            and entry["role"] in allowed_roles
            and isinstance(entry["dtype"], str)
            and bool(entry["dtype"])
            and isinstance(shape, list)
            and all(type(item) is int and item >= 0 for item in shape)
            and isinstance(stride, list)
            and len(stride) == len(shape)
            and all(type(item) is int and item >= 0 for item in stride)
            and type(entry["nbytes"]) is int
            and entry["nbytes"] >= 0
            and type(entry["storage_offset"]) is int
            and entry["storage_offset"] >= 0
            and type(entry["storage_nbytes"]) is int
            and entry["storage_nbytes"] >= 0
            and isinstance(entry["storage_group"], str)
            and bool(entry["storage_group"])
            and isinstance(storage_digest, str)
            and len(storage_digest) == 64
            and all(
                character in "0123456789abcdef"
                for character in storage_digest
            )
            and isinstance(digest, str)
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
        )

    if any(not valid_entry(entry) for entry in expected):
        _raise(
            "artifact",
            "artifact.schema",
            "Tensor manifest entries have invalid keys or JSON field types",
        )
    structural_keys = ("path", "role", "shape", "dtype", "nbytes")
    expected_structure = [
        {key: entry[key] for key in structural_keys}
        for entry in expected
    ]
    actual_structure = [
        {key: entry[key] for key in structural_keys}
        for entry in actual
    ]
    if expected_structure != actual_structure:
        _raise(
            "artifact",
            "artifact.schema",
            "Tensor key, role, shape, dtype, or byte length does not match manifest",
        )
    integrity_keys = (
        "stride",
        "storage_offset",
        "storage_nbytes",
        "storage_group",
        "storage_sha256",
        "sha256",
    )
    if [
        {key: entry[key] for key in integrity_keys} for entry in expected
    ] != [{key: entry[key] for key in integrity_keys} for entry in actual]:
        _raise(
            "artifact",
            "artifact.integrity",
            "Tensor content or storage/view identity does not match manifest",
        )
    actual_hash = _record_hash(raw)
    if payload["tensor_artifact_hash"] != actual_hash:
        _raise(
            "artifact",
            "artifact.integrity",
            "Tensor artifact hash does not match its manifest",
        )
    return tuple(MappingProxyType(dict(entry)) for entry in actual)


def _decode_payload(
    payload: Any,
    *,
    artifact: FixtureArtifact,
    reject_unreached_failure: bool = True,
) -> FixtureBundle:
    if not isinstance(payload, Mapping) or any(
        not isinstance(key, str) for key in payload
    ):
        _raise(
            "artifact",
            "artifact.schema",
            "fixture root must be a string-keyed mapping",
        )
    root = payload
    # These fields are the minimum root needed to authenticate the remaining
    # artifact.  Missing fields are schema failures because no claimed digest
    # exists to compare; when they exist, every integrity gate precedes the
    # complete artifact schema and Plan validators.
    integrity_fields = {
        "content_hash",
        "tensor_manifest",
        "tensor_artifact_hash",
        "logical_plan_bytes",
        "logical_plan_hash",
        "typed_plan_bytes",
        "typed_plan_hash",
    }
    if not integrity_fields.issubset(root):
        _raise(
            "artifact",
            "artifact.schema",
            "fixture root is missing fields required for integrity validation",
        )
    # Validate the weights-only-safe value domain before canonical digest
    # traversal so malformed nested mappings cannot leak sorting/serialization
    # exceptions outside the stable artifact envelope.
    _safe_value(root, path="fixture")
    actual_content_hash = _record_hash(
        {key: value for key, value in root.items() if key != "content_hash"}
    )
    if (
        not isinstance(root["content_hash"], str)
        or root["content_hash"] != actual_content_hash
    ):
        _raise(
            "artifact",
            "artifact.integrity",
            "fixture content hash does not match",
        )
    tensor_manifest = _verify_tensor_manifest(root)

    decoded_plan_records: Dict[str, Any] = {}
    for name in ("logical", "typed"):
        bytes_key = f"{name}_plan_bytes"
        hash_key = f"{name}_plan_hash"
        data = root[bytes_key]
        expected_hash = root[hash_key]
        if (
            not isinstance(data, bytes)
            or not isinstance(expected_hash, str)
            or hashlib.sha256(data).hexdigest() != expected_hash
        ):
            _raise(
                "artifact",
                "artifact.integrity",
                f"{name} Plan SHA-256 does not match canonical bytes",
            )
        record = _decode_json_bytes(data, context=f"{name} Plan bytes")
        if _canonical_json_bytes(record) != data:
            _raise(
                "artifact",
                "artifact.integrity",
                f"{name} Plan bytes are not canonical JSON",
            )
        decoded_plan_records[name] = record

    root = _exact_mapping(
        root,
        _ROOT_KEYS,
        context="fixture root",
        phase="artifact",
        code="artifact.schema",
    )
    if root["schema_version"] != FIXTURE_SCHEMA_VERSION:
        _raise(
            "artifact", "artifact.schema", "fixture schema version is incompatible"
        )
    if root["canonicalizer_id"] != PLAN_CANONICALIZER_ID:
        _raise(
            "artifact", "artifact.schema", "fixture canonicalizer ID is unsupported"
        )
    try:
        fixture_id = validate_stable_id(root["fixture_id"], kind="fixture")
    except ValueError as exc:
        _raise("artifact", "artifact.schema", str(exc))
    source = _validate_source(root["source"])
    expected = _validate_expected(root["expected"])
    routing = root["routing_classification"]
    if not isinstance(routing, str) or routing not in _ROUTING_CLASSES:
        _raise(
            "artifact",
            "artifact.schema",
            f"routing_classification must be one of {sorted(_ROUTING_CLASSES)!r}",
        )
    _exact_mapping(
        root["gradient"],
        _GRADIENT_KEYS,
        context="gradient",
        phase="artifact",
        code="artifact.schema",
    )
    parameter_schema = _validate_parameter_schema(
        root["parameter_schema"], root["logical_plan_hash"]
    )
    parameters = _as_string_tensor_mapping(root["parameters"], context="parameters")
    learnable_initial_state = _as_string_tensor_mapping(
        root["learnable_initial_state"], context="learnable_initial_state"
    )
    if learnable_initial_state:
        _raise(
            "artifact",
            "artifact.schema",
            "fixture v1 has no Plan schema for learnable initial state and requires an empty mapping",
        )

    logical_bytes = root["logical_plan_bytes"]
    plan = _decode_logical_plan(
        decoded_plan_records["logical"], plan_id=f"fixture:{fixture_id}"
    )
    if plan.canonical_bytes() != logical_bytes:
        _raise(
            "artifact",
            "artifact.integrity",
            "logical Plan bytes are not canonical",
        )
    typed_bytes = root["typed_plan_bytes"]
    typed_plan = _decode_typed_plan(
        plan,
        decoded_plan_records["typed"],
        expected_logical_hash=root["logical_plan_hash"],
    )
    if typed_plan.canonical_bytes() != typed_bytes:
        _raise(
            "artifact", "artifact.integrity", "typed Plan bytes are not canonical"
        )

    parameter_schema = _validate_parameter_schema(
        parameter_schema, root["logical_plan_hash"], plan
    )
    parameter_dtype = _dtype_from_binding(typed_plan, "parameter")
    if parameter_schema:
        schema_entries = {
            logical_parameter_tensor_key(entry["logical_key"]): entry
            for entry in parameter_schema["parameters"]
        }
        if set(parameters) != set(schema_entries):
            _raise(
                "artifact",
                "artifact.schema",
                "parameter Tensor keys do not exactly match parameter_schema",
            )
        for key, tensor in parameters.items():
            if tuple(tensor.shape) != tuple(schema_entries[key]["shape"]):
                _raise(
                    "artifact",
                    "artifact.schema",
                    f"parameter Tensor {key!r} shape does not match parameter_schema",
                )
    if any(
        not tensor.is_floating_point() or tensor.dtype != parameter_dtype
        for tensor in tuple(parameters.values())
        + tuple(learnable_initial_state.values())
    ):
        _raise(
            "artifact",
            "artifact.schema",
            "parameter and learnable initial-state tensors must use the bound parameter dtype",
        )
    _validate_distinct_tensor_storage(
        (("parameters", parameters), ("learnable_initial_state", learnable_initial_state))
    )
    # Whole-call input/mask/control schema precedes mutable state and position
    # continuity, matching the failure-stage contract.
    inputs, control = _validate_inputs_and_control(
        root["inputs"], root["control"], typed_plan
    )
    gradient = _validate_gradient(
        root["gradient"],
        hidden=inputs["hidden"],
        parameters=parameters,
        learnable_initial_state=learnable_initial_state,
    )
    try:
        initial_state = deserialize_state_store(
            root["initial_state"],
            device="cpu",
            dtype=_dtype_from_binding(typed_plan, "state"),
        )
    except CheckpointError as exc:
        _raise(
            "state",
            "state.schema",
            f"initial_state record is invalid: {exc}",
        )
    _validate_state_for_plan(initial_state, typed_plan)
    _validate_positions(inputs, control, initial_state)
    if reject_unreached_failure and expected["outcome"] == "failure":
        _raise(
            "artifact",
            "artifact.schema",
            "fixture declares failure but no loader-stage defect is reachable",
        )
    resolved_artifact = dataclasses.replace(
        artifact,
        content_hash=actual_content_hash,
        tensor_artifact_hash=root["tensor_artifact_hash"],
    )
    return FixtureBundle(
        fixture_id=fixture_id,
        typed_plan=typed_plan,
        source=MappingProxyType(dict(source)),
        parameter_schema=MappingProxyType(dict(parameter_schema)),
        inputs=MappingProxyType(dict(inputs)),
        parameters=parameters,
        learnable_initial_state=learnable_initial_state,
        initial_state=initial_state,
        control=MappingProxyType(dict(control)),
        expected=MappingProxyType(dict(expected)),
        gradient=MappingProxyType(dict(gradient)),
        routing_classification=routing,
        tensor_manifest=tensor_manifest,
        artifact=resolved_artifact,
    )


def _build_fixture_payload(
    *,
    fixture_id: str,
    typed_plan: TypedPlan,
    source: Mapping[str, Any],
    inputs: Mapping[str, Any],
    parameters: Mapping[str, Tensor],
    learnable_initial_state: Mapping[str, Tensor],
    initial_state: StateStore,
    control: Mapping[str, Any],
    expected: Mapping[str, Any],
    gradient: Mapping[str, Any],
    routing_classification: str,
    parameter_schema: Any,
) -> Dict[str, Any]:
    typed_plan.validate()
    try:
        validate_stable_id(fixture_id, kind="fixture")
    except ValueError as exc:
        _raise("artifact", "artifact.schema", str(exc))
    schema_value: Any
    if hasattr(parameter_schema, "canonical_dict"):
        schema_value = parameter_schema.canonical_dict()
    else:
        schema_value = parameter_schema
    source_parameters = _as_string_tensor_mapping(
        parameters, context="parameters"
    )
    source_learnable_state = _as_string_tensor_mapping(
        learnable_initial_state, context="learnable_initial_state"
    )
    # Check ownership before _safe_value creates independent CPU copies;
    # otherwise an illegal caller-side alias would be silently erased rather
    # than rejected by the parameter-group-free v1 schema.
    _validate_distinct_tensor_storage(
        (
            ("parameters", source_parameters),
            ("learnable_initial_state", source_learnable_state),
        )
    )
    _validate_state_for_plan(initial_state, typed_plan)
    try:
        state_record = serialize_state_store(initial_state)
    except CheckpointError as exc:
        _raise(
            "state",
            "state.schema",
            f"initial_state cannot be serialized: {exc}",
        )
    return {
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "fixture_id": fixture_id,
        "source": _safe_value(source, path="source"),
        "canonicalizer_id": PLAN_CANONICALIZER_ID,
        "logical_plan_bytes": typed_plan.logical_plan.canonical_bytes(),
        "logical_plan_hash": typed_plan.logical_hash(),
        "typed_plan_bytes": typed_plan.canonical_bytes(),
        "typed_plan_hash": typed_plan.typed_hash(),
        "parameter_schema": _safe_value(schema_value, path="parameter_schema"),
        "inputs": _safe_value(inputs, path="inputs"),
        "parameters": _safe_value(source_parameters, path="parameters"),
        "learnable_initial_state": _safe_value(
            source_learnable_state, path="learnable_initial_state"
        ),
        "initial_state": _safe_value(state_record, path="initial_state"),
        "control": _safe_value(control, path="control"),
        "expected": _safe_value(expected, path="expected"),
        "gradient": _safe_value(gradient, path="gradient"),
        "routing_classification": routing_classification,
    }


def _seal_payload(payload: Dict[str, Any]) -> None:
    for key in ("tensor_manifest", "tensor_artifact_hash", "content_hash"):
        payload.pop(key, None)
    tensor_entries = _build_tensor_manifest(payload)
    payload["tensor_manifest"] = {
        "schema_version": TENSOR_MANIFEST_SCHEMA_VERSION,
        "tensors": tensor_entries,
    }
    payload["tensor_artifact_hash"] = _record_hash(payload["tensor_manifest"])
    payload["content_hash"] = _record_hash(payload)


def _expected_envelope(payload: Mapping[str, Any]) -> Optional[FailureEnvelope]:
    expected = _validate_expected(payload.get("expected"))
    if expected["outcome"] == "success":
        return None
    return FailureEnvelope.from_dict(expected["error"])


def _preflight_payload(payload: Mapping[str, Any]) -> None:
    expected = _expected_envelope(payload)
    try:
        _decode_payload(
            payload,
            artifact=FixtureArtifact("", "", "", ""),
            reject_unreached_failure=False,
        )
    except FixtureError as exc:
        if expected is None:
            raise
        try:
            compare_failure_envelopes(expected, exc.envelope)
        except Exception as mismatch:
            _raise(
                "artifact",
                "artifact.schema",
                "negative fixture's declared envelope does not match its reachable failure: "
                f"{mismatch}",
            )
        return
    if expected is not None:
        _raise(
            "artifact",
            "artifact.schema",
            "fixture declares failure but passes every loader preflight gate",
        )


def _decode_container_bytes(data: bytes, *, context: str) -> Any:
    try:
        return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    except Exception as exc:
        _raise(
            "artifact",
            "artifact.integrity",
            f"cannot safely decode {context}: {type(exc).__name__}: {exc}",
        )


def _publish_payload(
    path: Union[str, os.PathLike[str]], payload: Dict[str, Any]
) -> FixtureArtifact:
    # The exact same semantic gates are run on the in-memory payload and on the
    # serialized bytes.  A negative bundle is publishable only when both reach
    # the failure envelope declared inside it.
    _preflight_payload(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"fixture destination already exists: {destination}")
    handle, temporary = tempfile.mkstemp(
        prefix=destination.name + ".",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    temporary_owned = True
    temporary_identity: Optional[Tuple[int, int]] = None
    published_identity: Optional[Tuple[int, int]] = None
    publication_complete = False
    try:
        initial = os.fstat(handle)
        temporary_identity = (initial.st_dev, initial.st_ino)
        with _closing_best_effort(os.fdopen(handle, "w+b")) as stream:
            handle = -1
            # Serialize through the already-open mkstemp descriptor.  Reopening
            # its pathname would permit a swapped symlink to overwrite an
            # unrelated file before publication.
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
            stream.seek(0)
            serialized = stream.read()
            probe = _decode_container_bytes(
                serialized, context="staged fixture"
            )
            _preflight_payload(probe)
            try:
                opened = os.fstat(stream.fileno())
                named = os.lstat(temporary)
            except OSError as exc:
                _raise(
                    "artifact",
                    "artifact.integrity",
                    f"cannot authenticate staged fixture pathname: {exc}",
                )
            if (
                not stat.S_ISREG(named.st_mode)
                or (opened.st_dev, opened.st_ino)
                != (named.st_dev, named.st_ino)
                or (opened.st_dev, opened.st_ino) != temporary_identity
            ):
                _raise(
                    "artifact",
                    "artifact.integrity",
                    "staged fixture pathname no longer names its open file",
                )
            # Hard-link publication is atomic and fails if a concurrent writer
            # has already claimed the immutable destination; it never replaces
            # existing data and keeps the authenticated inode alive.
            _link_open_file_no_replace(stream.fileno(), destination)
            published_identity = (opened.st_dev, opened.st_ino)
            published = os.lstat(destination)
            if (published.st_dev, published.st_ino) != published_identity:
                _raise(
                    "artifact",
                    "artifact.integrity",
                    "published fixture does not name the authenticated file",
                )
        # Delete the staging name before the commit barrier.  A successful
        # return persists both the immutable final link and the absence of a
        # temporary name.
        try:
            current_temporary = os.lstat(temporary)
        except FileNotFoundError:
            # Another actor consumed the name; the open descriptor was still
            # the publication source, and this process no longer owns a name
            # that it may clean up.
            temporary_owned = False
        else:
            if (
                current_temporary.st_dev,
                current_temporary.st_ino,
            ) == temporary_identity:
                os.unlink(temporary)
            # A different inode under the random name belongs to the actor
            # that recreated it and must survive our finalizer.
            temporary_owned = False
        _fsync_directory(destination.parent)
        publication_complete = True
    finally:
        if not publication_complete and published_identity is not None:
            # Do not remove a path that a concurrent actor replaced after the
            # hard-link operation.  Cleanup is attempted only while the final
            # name still identifies the inode authenticated above.
            try:
                current = os.lstat(destination)
                if (current.st_dev, current.st_ino) == published_identity:
                    os.unlink(destination)
                    _fsync_directory(destination.parent)
            except BaseException:
                # Preserve the publication failure.  An unknown or replaced
                # destination is deliberately left untouched.
                pass
        if handle >= 0:
            try:
                os.close(handle)
            except BaseException:
                pass
        if temporary_owned and temporary_identity is not None:
            try:
                current = os.lstat(temporary)
                if (current.st_dev, current.st_ino) == temporary_identity:
                    os.unlink(temporary)
            except BaseException:
                # Cleanup is identity-conditioned and best-effort; it must not
                # replace a publication failure or delete a recreated name.
                pass
    return FixtureArtifact(
        str(destination),
        hashlib.sha256(serialized).hexdigest(),
        payload["content_hash"],
        payload["tensor_artifact_hash"],
    )


def save_fixture_bundle(
    path: Union[str, os.PathLike[str]],
    *,
    fixture_id: str,
    typed_plan: TypedPlan,
    source: Mapping[str, Any],
    inputs: Mapping[str, Any],
    parameters: Mapping[str, Tensor],
    learnable_initial_state: Mapping[str, Tensor],
    initial_state: StateStore,
    control: Mapping[str, Any],
    expected: Mapping[str, Any],
    gradient: Mapping[str, Any],
    routing_classification: str,
    parameter_schema: Any,
) -> FixtureArtifact:
    """Validate and publish one immutable positive fixture bundle.

    A payload that merely declares ``expected.outcome='failure'`` is rejected
    unless a real loader-stage defect is present.  Use
    :func:`save_negative_fixture_bundle` to apply a named, reproducible defect.
    """

    payload = _build_fixture_payload(
        fixture_id=fixture_id,
        typed_plan=typed_plan,
        source=source,
        inputs=inputs,
        parameters=parameters,
        learnable_initial_state=learnable_initial_state,
        initial_state=initial_state,
        control=control,
        expected=expected,
        gradient=gradient,
        routing_classification=routing_classification,
        parameter_schema=parameter_schema,
    )
    _seal_payload(payload)
    return _publish_payload(path, payload)


def _canonical_record_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _apply_negative_mutation(payload: Dict[str, Any], mutation_kind: str) -> None:
    if mutation_kind not in _NEGATIVE_MUTATIONS:
        raise ValueError(
            f"unknown negative fixture mutation {mutation_kind!r}; "
            f"expected one of {sorted(_NEGATIVE_MUTATIONS)!r}"
        )
    source = payload["source"]
    if not isinstance(source, dict):
        _raise("artifact", "artifact.schema", "source must be mutable metadata")
    source["mutation_kind"] = mutation_kind
    if mutation_kind == "plan.topology.repeat-region-member":
        logical = _decode_json_bytes(
            payload["logical_plan_bytes"], context="base logical Plan bytes"
        )
        regions = logical.get("regions") if isinstance(logical, Mapping) else None
        if (
            not isinstance(regions, list)
            or not regions
            or not isinstance(regions[0], dict)
            or not isinstance(regions[0].get("node_ids"), list)
            or not regions[0]["node_ids"]
        ):
            raise ValueError("repeat-region-member mutation requires a nonempty region")
        regions[0]["node_ids"].append(regions[0]["node_ids"][0])
        logical_bytes = _canonical_record_bytes(logical)
        logical_hash = hashlib.sha256(logical_bytes).hexdigest()
        typed = _decode_json_bytes(
            payload["typed_plan_bytes"], context="base typed Plan bytes"
        )
        if not isinstance(typed, dict):
            raise ValueError("base typed Plan must be a mapping")
        typed["logical_plan_hash"] = logical_hash
        typed_bytes = _canonical_record_bytes(typed)
        payload["logical_plan_bytes"] = logical_bytes
        payload["logical_plan_hash"] = logical_hash
        payload["typed_plan_bytes"] = typed_bytes
        payload["typed_plan_hash"] = hashlib.sha256(typed_bytes).hexdigest()
        schema = payload["parameter_schema"]
        if not isinstance(schema, dict):
            raise ValueError("base parameter schema must be a mapping")
        schema["logical_plan_hash"] = logical_hash
        return
    if mutation_kind == "input.mask.lm-outside-execution":
        inputs = payload["inputs"]
        if not isinstance(inputs, dict):
            raise ValueError("mask mutation requires an input mapping")
        execution = inputs.get("execution_mask")
        lm_mask = inputs.get("lm_target_mask")
        if (
            not isinstance(execution, Tensor)
            or not isinstance(lm_mask, Tensor)
            or execution.ndim != 2
            or execution.numel() == 0
            or execution.shape != lm_mask.shape
        ):
            raise ValueError("mask mutation requires nonempty matching 2-D masks")
        execution = execution.clone()
        lm_mask = lm_mask.clone()
        execution.reshape(-1)[-1] = False
        lm_mask.reshape(-1)[-1] = True
        inputs["execution_mask"] = execution
        inputs["lm_target_mask"] = lm_mask
        return
    state = payload["initial_state"]
    entries = state.get("receiver_values") if isinstance(state, Mapping) else None
    tensor_entries = [
        item
        for item in entries or ()
        if isinstance(item, Mapping)
        and isinstance(item.get("payload"), Mapping)
        and item["payload"].get("kind") == "tensor"
        and isinstance(item["payload"].get("value"), Tensor)
    ]
    if len(tensor_entries) < 2:
        raise ValueError(
            "state owner-alias mutation requires two fixed-shape Tensor states"
        )
    first = tensor_entries[0]["payload"]["value"]
    second = tensor_entries[1]["payload"]["value"]
    backing = torch.empty(
        first.numel() + second.numel(), dtype=first.dtype, device="cpu"
    )
    backing[: first.numel()].copy_(first.reshape(-1))
    backing[first.numel() :].copy_(second.reshape(-1))
    tensor_entries[0]["payload"]["value"] = backing[: first.numel()].view(
        first.shape
    )
    tensor_entries[1]["payload"]["value"] = backing[first.numel() :].view(
        second.shape
    )


def save_negative_fixture_bundle(
    path: Union[str, os.PathLike[str]],
    *,
    mutation_kind: str,
    fixture_id: str,
    typed_plan: TypedPlan,
    source: Mapping[str, Any],
    inputs: Mapping[str, Any],
    parameters: Mapping[str, Tensor],
    learnable_initial_state: Mapping[str, Tensor],
    initial_state: StateStore,
    control: Mapping[str, Any],
    expected: Mapping[str, Any],
    gradient: Mapping[str, Any],
    routing_classification: str,
    parameter_schema: Any,
) -> FixtureArtifact:
    """Publish a valid base bundle after one named, authenticated mutation."""

    payload = _build_fixture_payload(
        fixture_id=fixture_id,
        typed_plan=typed_plan,
        source=source,
        inputs=inputs,
        parameters=parameters,
        learnable_initial_state=learnable_initial_state,
        initial_state=initial_state,
        control=control,
        expected=expected,
        gradient=gradient,
        routing_classification=routing_classification,
        parameter_schema=parameter_schema,
    )
    declared_envelope = _expected_envelope(payload)
    if declared_envelope is None:
        _raise(
            "artifact",
            "artifact.schema",
            "a negative fixture must declare expected.outcome='failure'",
        )
    declared_expected = payload["expected"]
    payload["expected"] = {"outcome": "success"}
    _seal_payload(payload)
    try:
        _decode_payload(
            payload,
            artifact=FixtureArtifact("", "", "", ""),
        )
    finally:
        payload["expected"] = declared_expected
    _apply_negative_mutation(payload, mutation_kind)
    _seal_payload(payload)
    return _publish_payload(path, payload)


def load_fixture_bundle(
    path: Union[str, os.PathLike[str]],
    *,
    expected_sha256: Optional[str] = None,
) -> FixtureBundle:
    """Authenticate and load a fixture without constructing an executor."""

    source_path = Path(path)
    try:
        data = source_path.read_bytes()
    except OSError as exc:
        _raise(
            "artifact",
            "artifact.integrity",
            f"cannot read fixture: {type(exc).__name__}: {exc}",
        )
    actual_sha256 = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        _raise(
            "artifact",
            "artifact.integrity",
            f"fixture SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}",
        )
    payload = _decode_container_bytes(data, context="fixture")
    return _decode_payload(
        payload,
        artifact=FixtureArtifact(str(source_path), actual_sha256, "", ""),
    )


__all__ = [
    "FIXTURE_SCHEMA_VERSION",
    "TENSOR_MANIFEST_SCHEMA_VERSION",
    "FixtureArtifact",
    "FixtureBundle",
    "FixtureError",
    "eager_parameter_tensors",
    "load_fixture_bundle",
    "logical_parameter_tensor_key",
    "save_fixture_bundle",
    "save_negative_fixture_bundle",
]
