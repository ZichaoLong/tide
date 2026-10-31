"""Phase-boundary observation only; no scheduling or workload changes."""
import resource
import sys
import torch
from tidegraph.runtime import synchronize

READY = "TIDE_WEIGHTED_RUNTIME_READY_V1"


def initialize(args, device):
    """Load the requested DSOs and execute a synchronized first device op."""
    if args.implementation != "python":
        from tidegraph.native_loader import load_native
        core = load_native(args.native_library, backend=device.type)
        if args.implementation == "resident":
            from tidegraph.resident_loader import load_resident
            load_resident(args.resident_library, core, component="weighted")
    probe = torch.ones((1, 1), device=device, dtype=getattr(torch, getattr(args, "dtype", "float32")))
    probe = probe @ probe
    synchronize(device)
    print(READY, flush=True)


class Memory:
    def __init__(self, device, peers=None):
        self.device = device
        self.devices = tuple(torch.device(d) for d in peers) if peers else (device,)
        self.allocator = getattr(torch, device.type) if device.type != "cpu" else None
        self.record = dict(schema="tide-weighted-memory-v1", phases=[],
            scope="process allocator only; excludes untracked vendor/driver memory; RSS is process-lifetime peak",
            sampling="outside step timers; peaks reset after runtime_ready, construction and warmup")

    def capture(self, phase, *, reset=False):
        from .storage import host_memory
        row = dict(phase=phase, process_lifetime_peak_rss_bytes=
                   resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
                   host_memory=host_memory())
        if self.allocator is not None:
            a, d = self.allocator, self.device
            row.update(device=str(d), allocated_bytes=a.memory_allocated(d),
                       reserved_bytes=a.memory_reserved(d),
                       peak_allocated_bytes=a.max_memory_allocated(d),
                       peak_reserved_bytes=a.max_memory_reserved(d))
            row["devices"] = [dict(device=str(peer), allocated_bytes=a.memory_allocated(peer),
                reserved_bytes=a.memory_reserved(peer), peak_allocated_bytes=a.max_memory_allocated(peer),
                peak_reserved_bytes=a.max_memory_reserved(peer)) for peer in self.devices]
            if reset:
                for peer in self.devices:a.reset_peak_memory_stats(peer)
        self.record["phases"].append(row)
        return self.record
