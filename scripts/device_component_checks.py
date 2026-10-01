"""Standalone device component gate names; these are not graph backend claims."""

CHECKS = {
    "peer-sharded-vjp": ("tide-sharded-vjp-check", ("float32", "float16")),
    "peer-shard-flow": ("tide-precision-flow-check", ("float32", "float16")),
    "peer-shard-control-flow": ("tide-precision-flow-check", ("float32", "float16")),
    "peer": ("tide-device-peer-check", ("float32", "float16")),
    "peer-flow": ("tide-precision-flow-check", ("float32", "float16")),
    "peer-control-flow": ("tide-precision-flow-check", ("float32", "float16")),
    "control": ("tide-device-control-check", ("float32",)),
    "failure": ("tide-device-failure-check", ("float32",)),
    "numerical": ("tide-device-numerical-check", ("float32", "float16")),
    "queue": ("tide-packed-queue-check", ("float32",)),
    "closure": ("tide-device-closure-check", ("float32",)),
    "transaction": ("tide-device-queue-check", ("float32", "float16")),
    "broadcast": ("tide-device-broadcast-check", ("float32", "float16")),
    "ready": ("tide-device-ready-check", ("float32", "float16")),
    "selector": ("tide-device-selector-check", ("float32",)),
    "content": ("tide-content-flow-check", ("float32",)),
    "window": ("tide-content-window-check", ("float32",)),
    "full": ("tide-packed-full-check", ("float32", "float16")),
    "packed-lh": ("tide-packed-lh-check", ("float32", "float16")),
    "aggregate-payload": ("tide-aggregate-payload-check", ("float32", "float16")),
    "state-read": ("tide-state-read-check", ("float32", "float16")),
    "precision-flow": ("tide-precision-flow-check", ("float32", "float16")),
    "attention-payload": ("tide-attention-payload-check", ("float32", "float16")),
    "sum": ("tide-packed-sum-check", ("float32", "float16")),
    "add": ("tide-device-add-check", ("float32",)),
    "graph-vjp": ("tide-device-graph-vjp-check", ("float32", "float16")),
    "parameter-vjp": ("tide-device-parameter-vjp-check", ("float32",)),
    "optimizer": ("tide-device-optimizer-check", ("float32", "float16")),
    "master-publication": ("tide-master-publication-check", ("float32", "float16")),
    "training-step": ("tide-device-training-step-check", ("float32",)),
    "retained": ("tide-device-retained-check", ("float32", "float16")),
    "extended-retained": ("tide-device-retained-check", ("float32", "float16")),
    "event-retained": ("tide-device-retained-check", ("float32", "float16")),
    "fiber-retained": ("tide-device-retained-check", ("float32", "float16")),
    "half-training": ("tide-resident-half-training-check", ("float16",)),
    "half-cache-training": ("tide-resident-half-training-check", ("float16",)),
    "reverse-links": ("tide-device-reverse-links-check", ("float32", "float16")),
    "full-vjp": ("tide-device-full-vjp-check", ("float32", "float16")),
    "state-vjp": ("tide-device-state-vjp-check", ("float32", "float16")),
    "clock": ("tide-device-clock-check", ("float32",)),
    "norm32": ("tide-device-norm-check", ("float32",)),
    "lh-full": ("tide-device-lh-full-check", ("float32",)),
    "origins": ("tide-device-origin-check", ("float32",)),
    "emission": ("tide-device-emission-check", ("float32", "float16")),
    "swiglu": ("tide-device-swiglu-check", ("float32", "float16")),
    "fiber": ("tide-device-fiber-check", ("float32",)),
    "fiber-pool": ("tide-device-fiber-pool-check", ("float32",)),
    "event-attention": ("tide-device-event-attention-check", ("float32",)),
    "attention-tile": ("tide-device-attention-tile-check", ("float32",)),
    "memory": ("tide-device-memory-check", ("float32",)),
    "event-batch": ("tide-device-event-batch-check", ("float32",)),
    "fiber-batch": ("tide-device-fiber-batch-check", ("float32",)),
    "aggregate": ("tide-device-aggregate-check", ("float32",)),
    "resident": ("tide-resident-check", ("float32",)),
    "resident-training": ("tide-resident-training-check", ("float32",)),
}
MARKERS = {
    "peer-sharded-vjp": "sharded-graph-vjp: passed",
    "peer-shard-flow": "peer-flow: passed",
    "peer-shard-control-flow": "peer-flow: passed",
    "peer": "device-peer: passed",
    "peer-flow": "peer-flow: passed",
    "peer-control-flow": "peer-flow: passed",
    "control": "device-control: passed", "numerical": "device-numerical: passed",
    "failure": "device-failure: passed",
    "queue": "packed-queue: passed", "closure": "device-closure: passed",
    "transaction": "device-queue: passed", "broadcast": "device-broadcast: passed",
    "ready": "device-ready: passed",
    "selector": "device-selector: passed",
    "content": "content-refusal: passed",
    "window": "device-window: passed",
    "full": "packed-full: passed",
    "packed-lh": "packed-lh: passed",
    "aggregate-payload": "aggregate-payload: passed",
    "state-read": "state-read: passed",
    "attention-payload": "attention-payload: passed",
    "precision-flow": "precision-flow: passed",
    "sum": "packed-sum: passed",
    "add": "device-add: passed",
    "graph-vjp": "device-graph-vjp: passed",
    "parameter-vjp": "device-parameter-vjp: passed",
    "optimizer": "device-optimizer: passed",
    "master-publication": "master-publication: passed",
    "training-step": "device-training-step: passed",
    "retained": "device-retained: passed",
    "extended-retained": "device-extended-retained: passed",
    "event-retained": "device-event-retained: passed",
    "fiber-retained": "device-fiber-retained: passed",
    "half-training": "resident-half-training: passed",
    "half-cache-training": "resident-half-cache-training: passed",
    "reverse-links": "device-reverse-links: passed",
    "full-vjp": "device-full-vjp: passed",
    "state-vjp": "device-state-vjp: passed",
    "clock": "device-clock: passed",
    "norm32": "device-norm32: passed",
    "lh-full": "device-lh-full: passed",
    "origins": "device-origins: passed",
    "emission": "device-emission: passed",
    "swiglu": "device-swiglu: passed",
    "fiber": "device-fiber: passed",
    "fiber-pool": "device-fiber-pool: passed",
    "event-attention": "device-event-attention: passed",
    "attention-tile": "device-attention-tile: passed",
    "memory": "device-memory: passed",
    "event-batch": "device-event-batch: passed",
    "fiber-batch": "device-fiber-batch: passed",
    "aggregate": "device-aggregate: passed",
    "resident": "public-resident: passed",
    "resident-training": "public-resident-training: passed",
}
KERNELS = {"master-publication": "tide_parameter_publish", "precision-flow": "tide_ready_pack", "attention-payload": "tide_attention_softmax", "aggregate-payload": "tide_aggregate_apply", "state-read": "tide_vector_state", "packed-lh": "tide_full_plan", "retained": "tide_window_bridge", "training-step": "tide_optimizer", "optimizer": "tide_optimizer", "parameter-vjp": "tide_parameter_vjp", "closure": "tide_closure", "transaction": "tide_queue_propose",
           "broadcast": "tide_broadcast_route", "ready": "tide_ready_pack", "selector": "tide_frame_select",
           "content": "tide_vector_sum", "window": "tide_vector_sum", "full": "tide_full_plan", "sum": "tide_vector_sum",
           "add": "tide_vector_state", "state-vjp": "tide_state_vjp", "full-vjp": "tide_full_vjp", "reverse-links": "tide_reverse_links", "graph-vjp": "tide_graph_reverse", "clock": "tide_vector_state", "norm32": "tide_vector_read",
           "lh-full": "tide_full_plan", "origins": "tide_sum_plan", "emission": "tide_emission_plan",
           "swiglu": "tide_full_plan", "fiber": "tide_fiber_payload", "fiber-pool": "tide_fiber_pool",
           "event-attention": "tide_event_payload", "attention-tile": "tide_attention_softmax", "memory": "tide_attention_softmax",
           "event-batch": "tide_event_plan", "fiber-batch": "tide_fiber_plan", "aggregate": "tide_aggregate_apply",
           "resident": "tide_ready_pack", "resident-training": "tide_optimizer"}

CHECKS["full-training"] = ("tide-resident-full-training-check", ("float32",))
MARKERS["full-training"] = "resident-full-training: passed"
KERNELS["full-training"] = "tide_extra_full"
KERNELS["extended-retained"] = "tide_window_bridge"
KERNELS["event-retained"] = "tide_event_reverse"
KERNELS["fiber-retained"] = "tide_fiber_reverse"
KERNELS["half-training"] = KERNELS["half-cache-training"] = "tide_optimizer"

CHECKS["extra-full-vjp"] = ("tide-device-extra-full-vjp-check", ("float32", "float16"))
MARKERS["extra-full-vjp"] = "device-extra-full-vjp: passed"
KERNELS["extra-full-vjp"] = "tide_extra_full"

CHECKS["aggregate-vjp"] = ("tide-device-aggregate-vjp-check", ("float32", "float16"))
MARKERS["aggregate-vjp"] = "device-aggregate-vjp: passed"
KERNELS["aggregate-vjp"] = "tide_aggregate_vjp"

CHECKS["aggregate-training"] = ("tide-resident-aggregate-training-check", ("float32",))
MARKERS["aggregate-training"] = "resident-aggregate-training: passed"
KERNELS["aggregate-training"] = "tide_aggregate_vjp"

CHECKS["control-vjp"] = ("tide-device-control-vjp-check", ("float32", "float16"))
MARKERS["control-vjp"] = "device-control-vjp: passed"
KERNELS["control-vjp"] = "tide_control_"
CHECKS["precision-control-flow"] = ("tide-precision-flow-check", ("float32", "float16"))
MARKERS["precision-control-flow"] = "scope=HST_SOFTP_inference"
KERNELS["precision-control-flow"] = "tide_emit_mix"

CHECKS["control-training"] = ("tide-resident-control-training-check", ("float32",))
MARKERS["control-training"] = "resident-control-training: passed"
KERNELS["control-training"] = "tide_control_"

CHECKS["attention-vjp"] = ("tide-device-attention-vjp-check", ("float32", "float16"))
MARKERS["attention-vjp"] = "device-attention-vjp: passed"
KERNELS["attention-vjp"] = "tide_attention_reverse_"

CHECKS["event-vjp"] = ("tide-device-event-vjp-check", ("float32", "float16"))
MARKERS["event-vjp"] = "device-event-vjp: passed"
KERNELS["event-vjp"] = "tide_event_reverse_"

CHECKS["event-training"] = ("tide-resident-event-training-check", ("float32",))
MARKERS["event-training"] = "resident-event-training: passed"
KERNELS["event-training"] = "tide_event_reverse_"

CHECKS["fiber-vjp"] = ("tide-device-fiber-vjp-check", ("float32", "float16"))
MARKERS["fiber-vjp"] = "device-fiber-vjp: passed"
KERNELS["fiber-vjp"] = "tide_fiber_vjp_"

CHECKS["fiber-reverse"] = ("tide-device-fiber-reverse-check", ("float32", "float16"))
MARKERS["fiber-reverse"] = "device-fiber-reverse: passed"
KERNELS["fiber-reverse"] = "tide_fiber_reverse_"

CHECKS["fiber-training"] = ("tide-resident-fiber-training-check", ("float32",))
MARKERS["fiber-training"] = "resident-fiber-training: passed"
KERNELS["fiber-training"] = "tide_fiber_reverse_"

KERNELS["peer-sharded-vjp"] = "tide_full_reverse_pack"
