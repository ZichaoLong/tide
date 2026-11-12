"""Independent DAG closure witnesses for the resident block metadata kernel.

CPU semantic-source checks do not certify hardware execution or WeightedFlow.
Numerical values are deliberately separate from message-presence booleans.
"""
import ctypes
from pathlib import Path
import random
import subprocess

import pytest
import torch


@pytest.fixture(scope="module")
def kernel(tmp_path_factory):
    root = Path(__file__).resolve().parents[1] / "tools/device_online"
    work = tmp_path_factory.mktemp("block_dispatch")
    source, binary = work / "adapter.cpp", work / "adapter.so"
    source.write_text('#define TIDE_KERNEL_CPU\n#include "tide_block_dispatch.cpp"\n')
    subprocess.run(["g++", "-std=c++17", "-shared", "-fPIC", "-O2",
                    "-I"+str(root/"cuda"), "-I"+str(root/"ascendc"),
                    "-I"+str(Path(torch.__file__).parent/"include"), str(source),
                    "-o", str(binary)], check=True)
    library = ctypes.CDLL(str(binary))
    call = library.tide_block_dispatch
    call.argtypes = [ctypes.c_void_p]*14 + [ctypes.c_int64]*6
    call.restype = None
    return call


def buffers(groups, edges, *, seeds=None, rows=4, journal=None, times=None):
    n = len(groups)
    tasks = torch.tensor([[g, i % 2, i // 2, (times or [0]*n)[i], 0]
                          for i, g in enumerate(groups)], dtype=torch.int64)
    fields = [tasks, torch.tensor(edges, dtype=torch.int64).reshape(-1, 3),
              torch.ones(n, dtype=torch.bool), torch.tensor(seeds or [False]*n),
              torch.ones(1, dtype=torch.int64), torch.zeros(n, 3, dtype=torch.int64),
              torch.zeros(n, 3, dtype=torch.int64), torch.full((rows,), n, dtype=torch.int64),
              torch.zeros(rows, dtype=torch.bool), torch.zeros(4, dtype=torch.int64),
              torch.zeros(4, dtype=torch.int64), torch.full((n if journal is None else journal, 3), -997, dtype=torch.int64),
              torch.zeros(1, dtype=torch.int32), torch.zeros(1, dtype=torch.int32)]
    return fields


def run(kernel, f, command):
    kernel(*(x.data_ptr() for x in f), command, f[0].shape[0], f[1].shape[0],
           int(f[0][:, 0].max())+1, f[7].numel(), f[11].shape[0])


def reference(groups, edges, seeds, output_present, unconditional=()):
    """Scalar recursive evaluation; never uses candidate batches or its state."""
    parents = [[] for _ in groups]
    for source, target, transmit in edges:
        parents[target].append((source, transmit))
    completed, present = {}, {}

    def visit(v):
        if v in completed:
            return
        for source, _ in parents[v]:
            visit(source)
        received = bool(seeds[v]) or any(transmit and present[source]
                                        for source, transmit in parents[v])
        executed = received or v in unconditional
        completed[v] = 2 if executed else 3
        present[v] = bool(executed and output_present[v])
    for v in range(len(groups)):
        visit(v)
    return [completed[v] for v in range(len(groups))], present


def drain(kernel, f, output_present):
    batches = []
    for _ in range(f[0].shape[0]+2):
        run(kernel, f, 1)
        assert not f[-1].item()
        count = int(f[9][0]); ids = f[7][:count].tolist()
        if ids:
            batches.append(ids)
        f[8][:count] = torch.tensor([output_present[v] for v in ids], dtype=torch.bool)
        run(kernel, f, 2)
        assert not f[-1].item()
        if not f[-2].item():
            return batches
    raise AssertionError("dispatcher did not terminate")


@pytest.mark.parametrize("rows", [1, 2, 8])
def test_online_absence_duplicate_edges_and_independent_regions(kernel, rows):
    groups = [0, 0, 1, 1, 0, 1]
    # Duplicate atoms are distinct causal obligations; a state-order edge does
    # not create a message. Nodes 0/1 are same-rank independent regions.
    edges = [(0, 2, 1), (0, 2, 1), (1, 3, 1), (2, 4, 1), (3, 4, 1), (4, 5, 0)]
    for seed in ([True, True, False, False, False, False], [False]*6):
        sends = [True, False, True, True, True, True]
        f = buffers(groups, edges, seeds=seed, rows=rows)
        run(kernel, f, 0); batches = drain(kernel, f, sends)
        expected, _ = reference(groups, edges, seed, sends)
        assert f[5][:, 0].tolist() == expected
        assert f[5][:, 1].eq(0).all()
        assert f[5][2, 2].item() == (2 if seed[0] else 0)
        assert int(f[10][2]) == expected.count(2)
        assert int(f[10][3]) == expected.count(3)
        if seed[0] and rows > 1:
            assert batches[0] == [0, 1]


def test_batch_and_time_axes_int64_feedback_and_reusable_window(kernel):
    # A positive-delay self-feedback is a time-expanded DAG; times are original
    # clocks, never compressed event counts. Two samples can execute together.
    base = 2**55+19
    groups = [0]*6
    edges = [(0, 2, 1), (1, 3, 1), (2, 4, 1), (3, 5, 1)]
    f = buffers(groups, edges, seeds=[True, True, False, False, False, False],
                times=[base, base, base+7, base+7, base+14, base+14], rows=4)
    for generation in range(1, 65):
        f[4].fill_(generation); run(kernel, f, 0)
        batches = drain(kernel, f, [True]*6)
        assert batches == [[0, 1], [2, 3], [4, 5]]
        assert f[10].tolist() == [generation, 3, 6, 0]
        assert f[0][:, 3].tolist() == [base, base, base+7, base+7, base+14, base+14]
    # Independent temporal blocks are eligible in one submission, not one/time.
    f = buffers([0]*6, [], seeds=[True]*6, times=[base+i for i in range(6)], rows=8)
    run(kernel, f, 0)
    assert drain(kernel, f, [True]*6) == [list(range(6))]


def test_zero_payload_presence_is_not_absence_and_guard_only_dependency(kernel):
    f = buffers([0, 0, 0], [(0, 1, 1), (0, 2, 0)], seeds=[True, False, True])
    run(kernel, f, 0)
    # The numerical stage may emit value 0 with presence True. No payload is
    # inspected by the scheduler; the guard-only edge cannot disable seed 2.
    assert drain(kernel, f, [True, True, True]) == [[0], [1, 2]]
    assert f[5][:, 0].tolist() == [2, 2, 2]


def test_unconditional_barrier_closes_missing_inputs(kernel):
    f = buffers([0, 1], [(0, 1, 0)])
    f[0][1, 4] = 1
    run(kernel, f, 0)
    assert drain(kernel, f, [False, True]) == [[1]]
    assert f[5][:, 0].tolist() == [3, 2]


def test_journal_capacity_and_numerical_failure_publish_no_prefix(kernel):
    f = buffers([0, 0, 0], [], seeds=[True, True, False], journal=1)
    run(kernel, f, 0)
    before = [f[i].clone() for i in (5, 10, 11)]
    run(kernel, f, 1); assert f[-1].item() == 71
    run(kernel, f, 2)
    assert all(torch.equal(f[i], x) for i, x in zip((5, 10, 11), before))
    f = buffers([0, 0], [], seeds=[True, False])
    run(kernel, f, 0); before = [f[i].clone() for i in (5, 10, 11)]
    run(kernel, f, 1); f[-1].fill_(21); run(kernel, f, 2)
    assert f[-1].item() == 21
    assert all(torch.equal(f[i], x) for i, x in zip((5, 10, 11), before))


def test_late_and_duplicate_completion_rejected(kernel):
    f = buffers([0], [], seeds=[True])
    run(kernel, f, 0); run(kernel, f, 1)
    f[4].add_(1); before = f[5].clone()
    run(kernel, f, 2)
    assert f[-1].item() == 73 and torch.equal(f[5], before)
    f = buffers([0], [], seeds=[False])
    run(kernel, f, 0); run(kernel, f, 1); run(kernel, f, 2)
    before = f[10].clone(); run(kernel, f, 2)
    assert f[-1].item() == 73 and torch.equal(f[10], before)


def test_cycle_and_invalid_descriptor_refused(kernel):
    f = buffers([0, 0], [(0, 1, 1), (1, 0, 1)], seeds=[True, False])
    run(kernel, f, 0); before = f[5].clone(); run(kernel, f, 1)
    assert f[-1].item() == 72 and torch.equal(f[5], before)
    f = buffers([0, 0], [(0, 2, 1)])
    before = f[5].clone(); run(kernel, f, 0)
    assert f[-1].item() == 70 and torch.equal(f[5], before)


def test_empty_reused_window_excludes_old_journal_rows(kernel):
    f = buffers([0, 0], [], seeds=[True, True])
    run(kernel, f, 0); drain(kernel, f, [True, True])
    f[3].zero_(); f[4].add_(1); run(kernel, f, 0)
    assert drain(kernel, f, [True, True]) == []
    assert f[10].tolist() == [2, 0, 0, 2]


def test_independent_formula_on_permuted_dags(kernel):
    rng = random.Random(70217)
    for _ in range(80):
        n = rng.randrange(2, 20); order = list(range(n)); rng.shuffle(order)
        groups = [rng.randrange(4) for _ in order]
        edges = []
        for i, source in enumerate(order):
            for target in order[i+1:]:
                if rng.random() < .15:
                    edge = (source, target, rng.randrange(2)); edges.append(edge)
                    if rng.random() < .2:
                        edges.append(edge)
        seeds = [rng.random() < .3 for _ in order]
        sends = [rng.random() < .6 for _ in order]
        unconditional = {i for i in order if rng.random() < .1}
        f = buffers(groups, edges, seeds=seeds, rows=rng.randrange(1, 8))
        for i in unconditional:
            f[0][i, 4] = 1
        run(kernel, f, 0); drain(kernel, f, sends)
        expected, _ = reference(groups, edges, seeds, sends, unconditional)
        assert f[5][:, 0].tolist() == expected
        assert f[5][:, 1].eq(0).all()
