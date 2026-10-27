"""CPU execution of the exact registered device sources; never a runtime fallback."""
import ctypes
from pathlib import Path
import subprocess
import sys
import torch


def build(directory):
    root = Path(__file__).resolve().parents[1]
    device = root / "tools/device_online"
    generated = directory / "generated"
    subprocess.run([sys.executable, str(device / "cuda/generate_kernels.py"), "--source",
                    str(device / "ascendc"), "--output", str(generated), "--backend", "cpu"], check=True)
    library = directory / "weighted.so"
    names = ("tide_weighted_allocate", "tide_weighted_allocation_vjp", "tide_weighted_allocation_status")
    subprocess.run(["g++", "-std=c++17", "-shared", "-fPIC", "-O2", "-ffp-contract=off",
                    "-I"+str(device / "cuda"), "-I"+str(device / "ascendc"),
                    "-I"+str(Path(torch.__file__).parent / "include"),
                    *(str(generated / (name+".cpp")) for name in names), "-o", str(library)], check=True)
    loaded = ctypes.CDLL(str(library))
    prefix = [ctypes.c_uint32, ctypes.c_void_p]
    pointers = [ctypes.c_void_p]*10
    loaded.tide_cuda_launch_tide_weighted_allocate.argtypes = prefix+pointers+[ctypes.c_int64]*3+[ctypes.c_float]
    loaded.tide_cuda_launch_tide_weighted_allocation_vjp.argtypes = prefix+pointers+[ctypes.c_int64]*3
    loaded.tide_cuda_launch_tide_weighted_allocation_status.argtypes = prefix+[ctypes.c_void_p]*2+[ctypes.c_int64]
    return loaded


class Run:
    def __init__(self, library, scores, caps, bounds, lengths, kind=0, alpha=1.):
        self.library = library
        self.scores, self.caps, self.bounds = [x.detach().contiguous().float() for x in (scores, caps, bounds)]
        self.lengths = torch.tensor(lengths, dtype=torch.int64)
        self.rows, self.width = scores.shape
        self.controls = torch.full_like(self.scores, float("nan"))
        self.slopes = torch.full((self.rows, self.width, 3), float("nan"))
        self.statistics = torch.full((self.rows, 2), float("nan"))
        self.branches = torch.zeros(self.rows, dtype=torch.int64)
        self.errors = torch.zeros(self.rows, dtype=torch.int32)
        self.error = torch.zeros(1, dtype=torch.int32)
        self.call("tide_weighted_allocate", [self.scores, self.caps, self.bounds, self.lengths,
                  self.controls, self.slopes, self.statistics, self.branches, self.errors, self.error],
                  self.rows, self.width, kind, alpha)
        self.finish()

    def call(self, name, tensors, *args):
        code = getattr(self.library, "tide_cuda_launch_"+name)(min(self.rows, 32), None,
                   *(x.data_ptr() for x in tensors), *args)
        assert code == 0

    def finish(self):
        code = self.library.tide_cuda_launch_tide_weighted_allocation_status(
            1, None, self.errors.data_ptr(), self.error.data_ptr(), self.rows)
        assert code == 0

    def backward(self, gradient, roots=None, has_shift=True):
        if roots is None:
            roots = [True]*self.rows
        gradient = gradient.contiguous().float()
        roots = torch.tensor(roots, dtype=torch.bool)
        vectors = torch.full((self.rows, 2, self.width), float("nan"))
        scalars = torch.full((self.rows, 4), float("nan"))
        connected = torch.zeros((self.rows, 6), dtype=torch.bool)
        self.call("tide_weighted_allocation_vjp", [self.lengths, self.branches, self.slopes, gradient, roots,
                  vectors, scalars, connected, self.errors, self.error], self.rows, self.width, int(has_shift))
        self.finish()
        return vectors, scalars, connected
