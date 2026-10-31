"""Explicit one-owner core-5 inventory; the full three-owner gate is separate."""

COMPONENTS = (
    "control", "numerical", "queue", "closure", "transaction", "broadcast", "ready", "selector",
    "weighted-allocation", "weighted-sending", "weighted-head", "weighted-region",
    "weighted-aggregate", "weighted-source", "weighted-parameters", "weighted-raw",
    "weighted-convolution", "weighted-projection", "weighted-kv-update", "weighted-pool",
    "weighted-kv-sum", "weighted-rotary", "weighted-kv-query", "weighted-residual", "weighted-kv-full",
    "weighted-describe", "weighted-recurrent", "weighted-flow", "weighted-flow-config",
    "weighted-stage", "weighted-reverse-route", "weighted-window", "weighted-window-config",
    "weighted-copy", "weighted-saved-window", "weighted-retained", "weighted-training",
    "optimizer", "master-publication",
)

PUBLIC_TESTS = (
    "test_resident_weighted_flow.py",
    "test_resident_weighted_storage_roots.py",
    "test_resident_weighted_training.py",
)
