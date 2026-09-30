"""Standalone device component gate names; these are not graph backend claims."""

CHECKS = {
    "control": ("tide-device-control-check", ("float32",)),
    "numerical": ("tide-device-numerical-check", ("float32", "float16")),
    "queue": ("tide-packed-queue-check", ("float32",)),
    "closure": ("tide-device-closure-check", ("float32",)),
    "transaction": ("tide-device-queue-check", ("float32", "float16")),
    "broadcast": ("tide-device-broadcast-check", ("float32", "float16")),
    "ready": ("tide-device-ready-check", ("float32", "float16")),
}
MARKERS = {
    "control": "device-control: passed", "numerical": "device-numerical: passed",
    "queue": "packed-queue: passed", "closure": "device-closure: passed",
    "transaction": "device-queue: passed", "broadcast": "device-broadcast: passed",
    "ready": "device-ready: passed",
}
KERNELS = {"closure": "tide_closure", "transaction": "tide_queue_propose",
           "broadcast": "tide_broadcast_route", "ready": "tide_ready_pack"}
