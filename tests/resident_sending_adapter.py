"""Same registered device sources, CPU semantic adapter only; no runtime fallback."""
import ctypes
from pathlib import Path
import subprocess
import sys
import torch


def build(directory):
    device = Path(__file__).resolve().parents[1] / "tools/device_online"
    generated = directory / "generated"
    subprocess.run([sys.executable, str(device / "cuda/generate_kernels.py"), "--source",
                    str(device / "ascendc"), "--output", str(generated), "--backend", "cpu"], check=True)
    names = ("tide_weighted_send", "tide_weighted_sending_vjp", "tide_weighted_allocation_status")
    path = directory / "sending.so"
    subprocess.run(["g++", "-std=c++17", "-shared", "-fPIC", "-O2", "-ffp-contract=off",
                    "-I"+str(device / "cuda"), "-I"+str(device / "ascendc"),
                    "-I"+str(Path(torch.__file__).parent / "include"),
                    *(str(generated / (n+".cpp")) for n in names), "-o", str(path)], check=True)
    lib = ctypes.CDLL(str(path));prefix = [ctypes.c_uint32, ctypes.c_void_p]
    lib.tide_cuda_launch_tide_weighted_send.argtypes = prefix+[ctypes.c_void_p]*12+[ctypes.c_int64]*5
    lib.tide_cuda_launch_tide_weighted_sending_vjp.argtypes = prefix+[ctypes.c_void_p]*11+[ctypes.c_int64]*5
    lib.tide_cuda_launch_tide_weighted_allocation_status.argtypes = prefix+[ctypes.c_void_p]*2+[ctypes.c_int64]
    return lib


class Run:
    def __init__(self, lib, scores, caps, received, parameters, ids, lengths, kind=0, base=0, k=0):
        self.lib, self.kind, self.base, self.k = lib, kind, base, k
        self.scores, self.caps, self.received, self.parameters = [x.detach().contiguous().float()
            for x in (scores, caps, received, parameters)]
        self.ids = torch.tensor(ids, dtype=torch.int64).contiguous()
        self.lengths = torch.tensor(lengths, dtype=torch.int64)
        self.rows, self.width = scores.shape
        self.controls = torch.full_like(scores, float("nan"))
        self.details = torch.full((self.rows, self.width, 4), float("nan"))
        self.statistics = torch.full((self.rows, 3), float("nan"))
        self.branches = torch.full((self.rows,), -42, dtype=torch.int64)
        self.errors = torch.zeros(self.rows, dtype=torch.int32);self.error = torch.zeros(1, dtype=torch.int32)
        self.call("tide_weighted_send", [self.scores, self.caps, self.received, self.parameters, self.ids,
            self.lengths, self.controls, self.details, self.statistics, self.branches, self.errors, self.error])
        self.finish()

    def call(self, name, tensors):
        assert getattr(self.lib, "tide_cuda_launch_"+name)(1, None, *(x.data_ptr() for x in tensors),
            self.rows, self.width, self.kind, self.base, self.k) == 0

    def finish(self):
        assert self.lib.tide_cuda_launch_tide_weighted_allocation_status(1, None,
            self.errors.data_ptr(), self.error.data_ptr(), self.rows) == 0

    def backward(self, gradient, roots=None):
        gradient = gradient.contiguous().float()
        roots = torch.tensor([True]*self.rows if roots is None else roots, dtype=torch.bool)
        vectors = torch.full((self.rows, 3, self.width), float("nan"))
        scalars = torch.full((self.rows, 2), float("nan"));flags = torch.zeros((self.rows, 5), dtype=torch.bool)
        self.call("tide_weighted_sending_vjp", [self.lengths, self.branches, self.details, self.statistics,
            gradient, roots, vectors, scalars, flags, self.errors, self.error])
        self.finish();return vectors, scalars, flags
