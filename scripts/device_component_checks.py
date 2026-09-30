"""Standalone device component gate names; these are not graph backend claims."""

CHECKS = {
    "control": ("tide-device-control-check", ("float32",)),
    "numerical": ("tide-device-numerical-check", ("float32", "float16")),
    "queue": ("tide-packed-queue-check", ("float32",)),
    "closure": ("tide-device-closure-check", ("float32",)),
    "transaction": ("tide-device-queue-check", ("float32", "float16")),
    "broadcast": ("tide-device-broadcast-check", ("float32", "float16")),
    "ready": ("tide-device-ready-check", ("float32", "float16")),
    "selector": ("tide-device-selector-check", ("float32",)),
    "content": ("tide-content-flow-check", ("float32",)),
    "full": ("tide-packed-full-check", ("float32",)),
}
MARKERS = {
    "control": "device-control: passed", "numerical": "device-numerical: passed",
    "queue": "packed-queue: passed", "closure": "device-closure: passed",
    "transaction": "device-queue: passed", "broadcast": "device-broadcast: passed",
    "ready": "device-ready: passed",
    "selector": "device-selector: passed",
    "content": "content-refusal: passed",
    "full": "packed-full: passed",
}
KERNELS = {"closure": "tide_closure", "transaction": "tide_queue_propose",
           "broadcast": "tide_broadcast_route", "ready": "tide_ready_pack", "selector": "tide_frame_select",
           "content": "tide_content_sum", "full": "tide_full_plan"}
