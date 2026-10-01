"""Benchmark-owned allocator counters at phase boundaries, not per-event reads."""
import resource
import sys
import torch
from tidegraph.runtime import resolve_device


class MemoryRecord:
    def __init__(self, devices):
        if not devices:
            raise ValueError("memory record needs at least one device")
        first, _ = resolve_device(str(devices[0]))
        resolved = [first, *(torch.device(d) for d in devices[1:])]
        if any(d.type != first.type for d in resolved):
            raise ValueError("consumer memory devices must use one backend")
        self.devices = resolved if first.type != "cpu" else []
        if any(d.index is None for d in self.devices) or len(set(self.devices)) != len(self.devices):
            raise ValueError("memory record needs distinct logical devices")
        self.phases = []
        self._reset()
        self.capture("initial", reset_peak=False)

    def _reset(self):
        for device in self.devices:
            getattr(torch, device.type).reset_peak_memory_stats(device)

    def capture(self, phase, reset_peak=True):
        devices = []
        for device in self.devices:
            stats = getattr(torch, device.type).memory_stats(device)
            devices.append(dict(device=str(device), allocated_bytes=stats["allocated_bytes.all.current"],
                peak_allocated_bytes=stats["allocated_bytes.all.peak"], reserved_bytes=stats["reserved_bytes.all.current"],
                peak_reserved_bytes=stats["reserved_bytes.all.peak"]))
        self.phases.append(dict(phase=phase, devices=devices,
            cpu_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == "darwin" else 1024)))
        if reset_peak:
            self._reset()

    def record(self):
        return dict(schema="tide-consumer-memory-v1", phases=self.phases,
            scope="per-process allocator per logical accelerator; excludes untracked vendor/driver memory; CPU RSS is process-lifetime peak",
            sampling="phase boundaries outside step timers; allocator peaks reset after construction and warmup; initial setup included in construction")
