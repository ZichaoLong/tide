"""Portable scalar evidence; no checkpoint tensors or runtime imports."""
from copy import deepcopy
from .records import fingerprint


DETAIL_FIELDS = (
    "schema", "state", "started", "finished", "start_window", "window", "phase",
    "identity", "runtime", "effective", "validation", "checkpoint", "checkpoint_publication", "admission",
    "metrics", "warmup", "observation", "construction_seconds", "validation_seconds",
    "total_wall_seconds", "timing_summary", "load_average_after", "error",
)


def details(record, resolved):
    """Retain requests, implementation identity, window costs and strict diagnostics.

    Repeated per-file identity maps are represented by their content digest and
    count. Aggregate package/build hashes and original source provenance remain
    intact. This is a reviewed record projection, never checkpoint retention or
    renewed qualification of an old binary.
    """
    result = deepcopy({key: record[key] for key in DETAIL_FIELDS if key in record})
    result["resolved_config"] = resolved

    def compact(value):
        if isinstance(value, dict):
            for key in tuple(value):
                child = value[key]
                if key in {"files_sha256", "source_files"} and isinstance(child, dict):
                    value[key] = dict(scope="per-file map identity; original map not embedded",
                                      count=len(child), sha256=fingerprint(child))
                else:
                    compact(child)
        elif isinstance(value, list):
            for child in value:
                compact(child)

    compact(result.get("identity", {}))
    return result
