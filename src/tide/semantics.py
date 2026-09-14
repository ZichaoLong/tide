"""Machine-readable binding from local Plans to the adopted semantics.

The logical Plan deliberately remains independent of documentation revisions:
the same graph and local formulas keep the same Plan hash when their semantic
authority is re-published without changing the instance.  Durable artifacts
carry the record defined here beside the Plan instead.

This module has no Torch dependency.  It can therefore be used by recording
tools before a device runtime is selected.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Union

from .plan import Plan


SEMANTIC_AUTHORITY_SCHEMA_VERSION = "tide.semantic-authority.v1"
SEMANTIC_CONTEXT_SCHEMA_VERSION = "tide.settlegraph.semantic-context.v1"
LOGICAL_SCHEDULE_SCHEMA_VERSION = "tide.settlegraph.logical-schedule.v1"
DEFAULT_NEXT_FORMULA_ID = "next.compute-snapshot.v1"
TRACE_SCHEMA_VERSION = "tide.settlegraph.trace.v1"
TRACE_PROJECTION_ID = "tide.settlegraph.trace-projection.lazy-proposal.v1"

UPSTREAM_LOCK_PATH = "docs/upstream/tide-core.lock.json"
UPSTREAM_ADOPTION_PATH = "docs/upstream-semantics.md"
LOCAL_SEMANTIC_AUTHORITY_PATH = "docs/experiment-semantics-and-naming.md"

# These hashes bind the source bytes used to interpret a durable artifact.
# The repository consistency test requires an intentional update whenever one
# of these authority documents changes.
UPSTREAM_LOCK_SHA256 = (
    "319994ddb2345f700d60882756bf68391fa22dc1ce699cc44e1635a1e2a81017"
)
UPSTREAM_ADOPTION_SHA256 = (
    "cf370edc2608eaf6c965652e3439bf7128c554d5f0f34e874f8fecc92c4d65d9"
)
LOCAL_SEMANTIC_AUTHORITY_SHA256 = (
    "1bbc8e37b76150932e719efcf8f00727aff24bafe35525d18a8138827878bfc2"
)

_UPSTREAM_LOCK = {
    "schema": "fractal-latcarf.upstream-semantics-lock.v1",
    "lock_version": 2,
    "upstream_repository": "https://github.com/ZichaoLong/ObsidianVault",
    "upstream_revision": "e529de212c605f7f417c3a1ce97e780a2ee59824",
    "semantic_version": "tide-core-3",
    "adopted_family": "SettleGraph",
    "anchor_path": "20-tide-decentralized-neural-network/semantics-anchor.md",
    "anchor_sha256": (
        "627a0603c58bbf69051b88b7a3fafaca9aad1ff4247b2a8516bb2ad5460cc341"
    ),
}


class SemanticContextError(ValueError):
    """A semantic authority, schedule, or Plan binding is malformed."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def semantic_authority_record() -> Dict[str, Any]:
    """Return a fresh, JSON-safe record of the adopted semantic authority."""

    return {
        "schema_version": SEMANTIC_AUTHORITY_SCHEMA_VERSION,
        "upstream_lock": dict(_UPSTREAM_LOCK),
        "documents": [
            {
                "role": "upstream-lock",
                "path": UPSTREAM_LOCK_PATH,
                "sha256": UPSTREAM_LOCK_SHA256,
            },
            {
                "role": "upstream-adoption",
                "path": UPSTREAM_ADOPTION_PATH,
                "sha256": UPSTREAM_ADOPTION_SHA256,
            },
            {
                "role": "local-instance-semantics",
                "path": LOCAL_SEMANTIC_AUTHORITY_PATH,
                "sha256": LOCAL_SEMANTIC_AUTHORITY_SHA256,
            },
        ],
    }


def _region_dependencies(plan: Plan) -> Dict[str, Set[str]]:
    """Derive semantic waits without using the executor's schedule helper."""

    region_of = {node.node_id: node.region_id for node in plan.nodes}
    dependencies: Dict[str, Set[str]] = {
        region.region_id: set(region.control_dependencies)
        for region in plan.regions
    }
    for edge in plan.edges:
        source = region_of[edge.source]
        target = region_of[edge.target]
        if source != target:
            dependencies[target].add(source)
    if plan.topology_kind == "hb":
        by_line: Dict[int, List[str]] = {}
        for region in plan.regions:
            assert isinstance(region.line, int)
            by_line.setdefault(region.line, []).append(region.region_id)
        for line in sorted(by_line):
            if line == 0:
                continue
            for target in by_line[line]:
                dependencies[target].update(by_line[line - 1])
    return dependencies


def logical_schedule_record(plan: Plan) -> Dict[str, Any]:
    """Return the fixed rank/stride schedule used by the local adaptation.

    General Plans use the longest path through data and explicit control
    dependencies.  HB Plans use ``line + 1`` as the region rank; Plan
    validation already guarantees that every data or control dependency goes
    from a shallower line to a deeper one.
    """

    plan.validate()
    dependencies = _region_dependencies(plan)
    if plan.topology_kind == "hb":
        region_ranks = {
            region.region_id: int(region.line) + 1 for region in plan.regions
        }
    else:
        unresolved = set(dependencies)
        region_ranks: Dict[str, int] = {}
        while unresolved:
            ready = sorted(
                region_id
                for region_id in unresolved
                if dependencies[region_id] <= set(region_ranks)
            )
            if not ready:  # Defensive: Plan.validate() rejects this first.
                raise SemanticContextError(
                    "cannot derive logical ranks from a cyclic region schedule"
                )
            for region_id in ready:
                parents = dependencies[region_id]
                region_ranks[region_id] = (
                    1 + max(region_ranks[parent] for parent in parents)
                    if parents
                    else 1
                )
                unresolved.remove(region_id)

    node_ranks = {
        node.node_id: region_ranks[node.region_id] for node in plan.nodes
    }
    output_rank = 1 + max(node_ranks.values())
    step_stride = output_rank + 1
    return {
        "schema_version": LOGICAL_SCHEDULE_SCHEMA_VERSION,
        "region_ranks": [
            {"region_id": region_id, "rank": region_ranks[region_id]}
            for region_id in sorted(region_ranks)
        ],
        "node_ranks": [
            {"node_id": node_id, "rank": node_ranks[node_id]}
            for node_id in sorted(node_ranks)
        ],
        "edge_delays": [
            {
                "edge_id": edge.edge_id,
                "delay": node_ranks[edge.target] - node_ranks[edge.source],
            }
            for edge in sorted(plan.edges, key=lambda item: item.edge_id)
        ],
        "output_rank": output_rank,
        "step_stride": step_stride,
    }


def semantic_context_for_plan(plan: Plan) -> Dict[str, Any]:
    """Bind one normalized Plan to authority, schedule, Next, and projection."""

    plan.validate()
    return {
        "schema_version": SEMANTIC_CONTEXT_SCHEMA_VERSION,
        "logical_plan_hash": plan.canonical_hash(),
        "authority": semantic_authority_record(),
        "logical_schedule": logical_schedule_record(plan),
        "next_formula_by_node": [
            {"node_id": node.node_id, "formula_id": DEFAULT_NEXT_FORMULA_ID}
            for node in sorted(plan.nodes, key=lambda item: item.node_id)
        ],
        "selector_history_formula_by_region": [
            {
                "region_id": region.region_id,
                "formula_id": region.selector_history["formula_id"],
            }
            for region in sorted(plan.regions, key=lambda item: item.region_id)
        ],
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "trace_projection_id": TRACE_PROJECTION_ID,
    }


def semantic_context_hash(plan: Plan) -> str:
    """Return the canonical SHA-256 identity of one Plan semantic context."""

    return hashlib.sha256(_canonical_bytes(semantic_context_for_plan(plan))).hexdigest()


def validate_semantic_context(plan: Plan, value: Any) -> Dict[str, Any]:
    """Require an exact current semantic context for ``plan``."""

    if not isinstance(value, Mapping):
        raise SemanticContextError("semantic_context must be a mapping")
    actual = dict(value)
    expected = semantic_context_for_plan(plan)
    if actual != expected:
        raise SemanticContextError(
            "semantic_context does not match the Plan and adopted authority"
        )
    return actual


def validate_repository_semantic_sources(
    repository_root: Union[str, Path],
) -> Dict[str, Any]:
    """Check that tracked authority bytes and the machine binding agree.

    The upstream anchor itself is intentionally not required to exist in this
    checkout.  Its immutable revision and raw-byte digest are authenticated by
    the lock; a separate upstream checkout may verify those bytes when it is
    available.
    """

    root = Path(repository_root)
    expected_documents: Tuple[Tuple[str, str], ...] = (
        (UPSTREAM_LOCK_PATH, UPSTREAM_LOCK_SHA256),
        (UPSTREAM_ADOPTION_PATH, UPSTREAM_ADOPTION_SHA256),
        (LOCAL_SEMANTIC_AUTHORITY_PATH, LOCAL_SEMANTIC_AUTHORITY_SHA256),
    )
    for relative, expected_hash in expected_documents:
        path = root / relative
        if not path.is_file():
            raise SemanticContextError(
                f"semantic authority document is missing: {relative}"
            )
        actual_hash = _file_sha256(path)
        if actual_hash != expected_hash:
            raise SemanticContextError(
                f"semantic authority hash mismatch for {relative}: "
                f"expected {expected_hash}, got {actual_hash}"
            )

    try:
        lock = json.loads((root / UPSTREAM_LOCK_PATH).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SemanticContextError(
            f"cannot decode the upstream semantic lock: {type(exc).__name__}: {exc}"
        ) from exc
    if lock != _UPSTREAM_LOCK:
        raise SemanticContextError(
            "upstream semantic lock fields do not match the machine binding"
        )
    adoption_text = (root / UPSTREAM_ADOPTION_PATH).read_text(encoding="utf-8")
    for required in (
        _UPSTREAM_LOCK["upstream_revision"],
        _UPSTREAM_LOCK["semantic_version"],
        _UPSTREAM_LOCK["anchor_sha256"],
    ):
        if required not in adoption_text:
            raise SemanticContextError(
                "upstream adoption document does not quote the locked identity"
            )
    return semantic_authority_record()


__all__ = [
    "DEFAULT_NEXT_FORMULA_ID",
    "LOCAL_SEMANTIC_AUTHORITY_PATH",
    "LOCAL_SEMANTIC_AUTHORITY_SHA256",
    "LOGICAL_SCHEDULE_SCHEMA_VERSION",
    "SEMANTIC_AUTHORITY_SCHEMA_VERSION",
    "SEMANTIC_CONTEXT_SCHEMA_VERSION",
    "SemanticContextError",
    "TRACE_PROJECTION_ID",
    "TRACE_SCHEMA_VERSION",
    "UPSTREAM_ADOPTION_PATH",
    "UPSTREAM_ADOPTION_SHA256",
    "UPSTREAM_LOCK_PATH",
    "UPSTREAM_LOCK_SHA256",
    "logical_schedule_record",
    "semantic_authority_record",
    "semantic_context_for_plan",
    "semantic_context_hash",
    "validate_repository_semantic_sources",
    "validate_semantic_context",
]
