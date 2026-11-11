"""Single-bank guard semantic source; real GEMM/VJP qualification is separate."""
import ctypes
from pathlib import Path
import subprocess
import sys
import pytest
import torch


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    device = Path(__file__).resolve().parents[1] / "tools/device_online"
    directory = tmp_path_factory.mktemp("shared-projection")
    generated = directory / "generated"
    subprocess.run([sys.executable, str(device/"cuda/generate_kernels.py"), "--source", str(device/"ascendc"),
                    "--output", str(generated), "--backend", "cpu"], check=True)
    path = directory/"guard.so"
    subprocess.run(["g++", "-std=c++17", "-shared", "-fPIC", "-O2", "-ffp-contract=off",
                    "-I"+str(device/"cuda"), "-I"+str(device/"ascendc"),
                    "-I"+str(Path(torch.__file__).parent/"include"),
                    str(generated/"tide_weighted_projection_shared.cpp"), "-o", str(path)], check=True)
    lib = ctypes.CDLL(str(path))
    run = lib.tide_cuda_launch_tide_weighted_projection_shared
    run.argtypes = [ctypes.c_uint32, ctypes.c_void_p] + [ctypes.c_void_p]*9 + [ctypes.c_int64]*5
    return run


def call(run, owners, roots, source, bias, saved, saved_bias, dx, connected, error, mode):
    assert run(1, None, *(t.data_ptr() for t in
        (owners, roots, source, bias, saved, saved_bias, dx, connected, error)),
        owners.numel(), saved.shape[-1], saved.shape[-2], int(bias.numel() > 0), mode) == 0


@pytest.mark.parametrize("bias", [False, True])
def test_snapshot_actual_use_and_absent_poison(library, bias):
    source = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    b = torch.arange(3, dtype=torch.float32).reshape(1, 3) if bias else torch.empty(0)
    saved, sb = torch.full_like(source, 42), torch.full_like(b, 42)
    ids = torch.tensor([-1, 0, -1]); roots = torch.zeros(3, dtype=torch.bool)
    connected = torch.zeros(1, dtype=torch.bool); error = torch.zeros(1, dtype=torch.int32)
    dx = torch.full((3, 4), float("nan"))
    call(library, ids, roots, source, b, saved, sb, dx, connected, error, 0)
    assert error.item() == 0 and torch.equal(saved, source) and torch.equal(sb, b)
    source.fill_(float("nan")); b.fill_(float("nan"))
    assert saved.isfinite().all() and sb.isfinite().all()
    ids.fill_(-1)
    call(library, ids, roots, source, b, saved, sb, dx, connected, error, 0)
    assert error.item() == 0 and saved.eq(0).all() and sb.eq(0).all()
    ids[0] = 0
    call(library, ids, roots, source, b, saved, sb, dx, connected, error, 0)
    assert error.item() == 20


def test_transpose_none_zero_finite_and_sticky_error(library):
    ids = torch.tensor([0, -1, 0]); roots = torch.tensor([True, False, False])
    matrix, bias = torch.zeros(1, 3, 4), torch.zeros(1, 3)
    dx = torch.full((3, 4), float("nan")); dx[0].zero_()
    connected = torch.zeros(1, dtype=torch.bool); error = torch.zeros(1, dtype=torch.int32)
    call(library, ids, roots, matrix, bias, matrix, bias, dx, connected, error, 1)
    assert error.item() == 0 and connected.item() and dx.eq(0).all()
    roots.zero_(); matrix.fill_(float("nan")); bias.fill_(float("nan")); dx.fill_(float("nan"))
    call(library, ids, roots, matrix, bias, matrix, bias, dx, connected, error, 1)
    assert error.item() == 0 and not connected.item() and dx.eq(0).all() and matrix.eq(0).all() and bias.eq(0).all()
    roots[0] = True; matrix[0, 0, 0] = float("inf")
    call(library, ids, roots, matrix, bias, matrix, bias, dx, connected, error, 1)
    assert error.item() == 23
    matrix.fill_(17); dx.fill_(17)
    call(library, ids, roots, matrix, bias, matrix, bias, dx, connected, error, 1)
    assert error.item() == 23 and matrix.eq(17).all() and dx.eq(17).all()
