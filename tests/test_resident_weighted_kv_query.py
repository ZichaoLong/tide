"""KV §3.3: actual input use and four independently rooted Mem returns.

Numerical formula/parameter/raw VJP checks run in weighted_kv_query_check.cpp.
These CPU checks exercise the device metadata kernels against eager dependency
graphs, including empty short Attention after numerical decay underflow.
"""
import ctypes
from pathlib import Path
import subprocess
import pytest
import torch


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    root = Path(__file__).resolve().parents[1] / "tools/device_online"
    folder = tmp_path_factory.mktemp("kv-query")
    source, output = folder / "adapter.cpp", folder / "adapter.so"
    source.write_text('''#define TIDE_KERNEL_CPU
#include "weighted_kv_query_math.h"
extern "C" void plan(float* w,int64_t* n,uint8_t* p,int64_t* owner,float* u,
 float* clean,int64_t* q,int64_t* entries,int32_t* error,
 int64_t rows,int64_t capacity,int64_t width,int64_t banks){
 tide_weighted_query_device::plan(w,n,p,owner,u,clean,q,entries,error,rows,capacity,width,banks);
}
extern "C" void roots(uint8_t* pool,int64_t* owners,uint8_t* q,uint8_t* k,
 uint8_t* v,uint8_t* w,int32_t* error,int64_t rows,int64_t capacity){
 tide_weighted_query_device::roots(pool,owners,q,k,v,w,error,rows,capacity);
}
''')
    subprocess.run(["g++", "-std=c++17", "-shared", "-fPIC", "-O2",
                    "-I"+str(root/"cuda"), "-I"+str(root/"ascendc"),
                    "-I"+str(Path(torch.__file__).parent/"include"),
                    str(source), "-o", str(output)], check=True)
    lib = ctypes.CDLL(str(output))
    lib.plan.argtypes = [ctypes.c_void_p]*9 + [ctypes.c_int64]*4
    lib.roots.argtypes = [ctypes.c_void_p]*7 + [ctypes.c_int64]*2
    return lib


def plan_inputs():
    weights = torch.tensor([[[0., 0., 0.], [.3, 0., .2]],
                            [[0., 0., 0.], [0., 0., 0.]],
                            [[float("nan")]*3]*2])
    lengths = torch.tensor([3, 2, -99]); present = torch.tensor([True, True, False])
    owners = torch.tensor([1, -99, -99])
    u = torch.tensor([[.5, -.4], [float("nan")]*2, [float("nan")]*2])
    clean = torch.full_like(u, float("nan"))
    query = torch.full_like(owners, -8); entries = torch.full((3, 3), -8)
    error = torch.zeros(1, dtype=torch.int32)
    return weights, lengths, present, owners, u, clean, query, entries, error


def test_only_positive_long_memory_uses_query_and_parameter_owner(library):
    args = plan_inputs()
    library.plan(*(x.data_ptr() for x in args), 3, 3, 2, 2)
    *_, clean, query, entries, error = args
    assert error.item() == 0
    assert query.tolist() == [1, -1, -1]
    assert entries.tolist() == [[1, -1, 1], [-1]*3, [-1]*3]
    torch.testing.assert_close(clean, torch.tensor([[.5, -.4], [0., 0.], [0., 0.]]))


@pytest.mark.parametrize("bad_owner", [True, False])
def test_actual_query_use_refuses_invalid_owner_or_content(library, bad_owner):
    args = plan_inputs()
    if bad_owner:
        args[3][0] = 2
    else:
        args[4][0, 0] = float("nan")
    library.plan(*(x.data_ptr() for x in args), 3, 3, 2, 2)
    assert args[-1].item() == (2 if bad_owner else 20)


@pytest.mark.parametrize("root", range(4))
def test_roots_match_independent_attention_dependencies(library, root):
    # The short channel has underflowed; the long channel still has two keys.
    # Parameter roles and raw fields are separate leaves, even at zero bars.
    query = torch.tensor([.2, -.3], requires_grad=True)
    key = torch.tensor([[.3, .1], [-.2, .4]], requires_grad=True)
    value = torch.tensor([[.5, -.1], [.7, .4]], requires_grad=True)
    write = torch.tensor([.4, .6], requires_grad=True)
    outputs, flags = [], torch.zeros((1, 2, 3), dtype=torch.bool)
    for channel in range(2):
        weight = write * (0. if channel == 0 else 1.)
        logits = key @ query
        total = weight.sum()
        positive = weight > 0
        vector = (total.clamp_max(1) *
                  ((logits[positive]+weight[positive].log()).softmax(0).unsqueeze(1)*value[positive]).sum(0)
                  if positive.any() else torch.zeros(2))
        outputs.append((vector, total))
        output = vector if root < 2 else total
        if root % 2 == channel and output.requires_grad:
            grads = torch.autograd.grad(output, (weight, value, logits),
                                        torch.zeros_like(output), allow_unused=True, retain_graph=True)
            flags[0, channel] = torch.tensor([g is not None for g in grads])
    result = outputs[root % 2][root // 2]
    expected = (torch.autograd.grad(result, (query, key, value, write),
                                   torch.zeros_like(result), allow_unused=True)
                if result.requires_grad else (None,)*4)
    owners = torch.tensor([[1, 1, -1]])
    q = torch.zeros(1, dtype=torch.bool); k = torch.zeros((1, 3), dtype=torch.bool)
    v = torch.zeros_like(k); w = torch.zeros((1, 5), dtype=torch.bool)
    error = torch.zeros(1, dtype=torch.int32)
    library.roots(*(x.data_ptr() for x in (flags, owners, q, k, v, w, error)), 1, 3)
    assert not error.item()
    assert q.item() == (expected[0] is not None)
    assert k[0].tolist() == [expected[1] is not None]*2 + [False]
    assert v[0].tolist() == [expected[2] is not None]*2 + [False]
    assert w[0, :3].tolist() == [expected[1] is not None, expected[2] is not None, False]
    assert w[0, 3:].any().item() == (expected[3] is not None)
