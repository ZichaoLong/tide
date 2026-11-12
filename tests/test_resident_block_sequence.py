"""Exact KV raw sequence and transpose, independently from candidate scheduling.

CPU executes the device metadata semantic source. Vendor projection and full
local numerical composition receive a separate standalone device check.
"""
import ctypes
from pathlib import Path
import subprocess

import pytest
import torch


@pytest.fixture(scope="module")
def kernel(tmp_path_factory):
    root = Path(__file__).resolve().parents[1]/"tools/device_online"
    work = tmp_path_factory.mktemp("block_sequence")
    src, lib = work/"adapter.cpp", work/"adapter.so"
    src.write_text('#define TIDE_KERNEL_CPU\n#include "tide_block_kv_sequence.cpp"\n'
                   '#include "tide_block_kv_sequence_reverse.cpp"\n')
    subprocess.run(["g++", "-std=c++17", "-shared", "-fPIC", "-O2",
                    "-I"+str(root/"cuda"), "-I"+str(root/"ascendc"),
                    "-I"+str(Path(torch.__file__).parent/"include"), str(src), "-o", str(lib)], check=True)
    dll = ctypes.CDLL(str(lib))
    dll.tide_block_kv_sequence.argtypes = [ctypes.c_void_p]*14+[ctypes.c_int64]*8
    dll.tide_block_kv_sequence_reverse.argtypes = [ctypes.c_void_p]*16+[ctypes.c_int64]*7
    return dll


class Fixture:
    batch, length, nodes, capacity, width, write_width = 2, 4, 2, 3, 21, 9

    def __init__(self, horizon=7):
        self.horizon = horizon
        b, l, n, c = self.batch, self.length, self.nodes, self.capacity
        self.rows, self.owners = b*l*n, b*n
        self.fields = torch.tensor([[0, 0, 3, 0], [3, 3, 2, 1], [9, 5, 2, 1],
                                    [15, 7, 1, 1], [18, 8, 1, 1]])
        self.initial = torch.zeros(self.owners, self.width)
        self.initial_i = torch.zeros(self.owners, 4+c, dtype=torch.int64)
        self.write = torch.zeros(self.rows, self.write_width)
        self.write_i = torch.zeros(self.rows, 5, dtype=torch.int64)
        self.times = torch.zeros(b, l, n, dtype=torch.int64)
        self.initial_fields, self.write_fields = [], []
        for owner in range(self.owners):
            base = 2**54+17+(owner//n)*100+owner % n
            stamps = [] if owner == 1 else [base-6, base-2]
            self.initial_i[owner, :4] = torch.tensor([stamps[-1] if stamps else -1, len(stamps), len(stamps), 0])
            self.initial_i[owner, 4:4+len(stamps)] = torch.tensor(stamps, dtype=torch.int64)
            values = []
            for field, (off, _, size, ragged) in enumerate(self.fields.tolist()):
                count = len(stamps) if ragged else 1
                x = torch.arange(count*size, dtype=torch.float32).reshape(count, size)*.03+.1+owner*.01
                if field == 0:
                    x.zero_()
                values.append(x.requires_grad_())
                self.initial[owner, off:off+x.numel()] = x.detach().flatten()
            self.initial_fields.append(values)
        for row in range(self.rows):
            sample, step, node = row//(l*n), row//n % l, row % n
            time = 2**54+17+sample*100+node+[0, 2, 5, 11][step]
            self.times[sample, step, node] = time
            positive = (row % 5 != 1 and row != 3)
            self.write_i[row] = torch.tensor([time, 1, 1, 0, time] if positive else [-1, 0, 0, 0, 0])
            values = []
            for field, (_, off, size, _) in enumerate(self.fields.tolist()):
                x = torch.arange(size, dtype=torch.float32).reshape(1, size)*.02+.2+row*.005
                if field == 0:
                    x.zero_()
                values.append(x.requires_grad_())
                self.write[row, off:off+size] = x.detach().flatten()
            self.write_fields.append(values)
        self.old = torch.empty(self.rows, self.width)
        self.old_i = torch.empty(self.rows, 4+c, dtype=torch.int64)
        self.proposal, self.proposal_i = torch.empty_like(self.old), torch.empty_like(self.old_i)
        self.final, self.final_i = torch.empty_like(self.initial), torch.empty_like(self.initial_i)
        self.mapping = torch.empty(2, self.rows, c, dtype=torch.int64)
        self.error = torch.zeros(1, dtype=torch.int32)

    def run(self, kernel):
        arrays = (self.fields, self.initial, self.initial_i, self.write, self.write_i,
                  self.times, self.old, self.old_i, self.proposal, self.proposal_i,
                  self.final, self.final_i, self.mapping, self.error)
        kernel.tide_block_kv_sequence(*(x.data_ptr() for x in arrays), self.batch, self.length,
                                     self.nodes, self.width, self.write_width, self.capacity, 5, self.horizon)

    def reference(self):
        old, proposal, final = [None]*self.rows, [None]*self.rows, []
        for owner in range(self.owners):
            sample, node = divmod(owner, self.nodes)
            count = int(self.initial_i[owner, 2])
            stamps = self.initial_i[owner, 4:4+count].tolist()
            fields = self.initial_fields[owner]
            last, observed = self.initial_i[owner, :2].tolist()
            for step in range(self.length):
                row = (sample*self.length+step)*self.nodes+node
                old[row] = (list(stamps), last, observed, fields)
                if self.write_i[row, 1]:
                    time = int(self.times.flatten()[row])
                    retain = torch.tensor([i for i, stamp in enumerate(stamps)
                                           if not self.horizon or time-stamp < self.horizon], dtype=torch.int64)
                    stamps = [stamps[i] for i in retain.tolist()]+[time]
                    fields = [fields[0]]+[torch.cat((fields[j][retain], self.write_fields[row][j])) for j in range(1, 5)]
                    last, observed = time, observed+1
                proposal[row] = (list(stamps), last, observed, fields)
            final.append(proposal[row])
        return old, proposal, final

    def pack(self, state):
        out = torch.zeros(self.width)
        stamps, last, observed, fields = state
        for (offset, _, _, _), value in zip(self.fields.tolist(), fields):
            out[offset:offset+value.numel()] = value.detach().flatten()
        meta = torch.tensor([last, observed, len(stamps), 0]+stamps+[0]*(self.capacity-len(stamps)))
        return out, meta


def test_sequence_actual_snapshots_and_two_axis_isolation(kernel):
    f = Fixture(); f.run(kernel)
    assert not f.error.item()
    for (values, metadata), reference in zip(((f.old, f.old_i), (f.proposal, f.proposal_i), (f.final, f.final_i)), f.reference()):
        for row, state in enumerate(reference):
            actual, integer = f.pack(state)
            assert torch.equal(values[row], actual)
            assert torch.equal(metadata[row], integer)
    # Different sequences have disjoint histories even with the same node ID.
    assert f.final_i[0, 0] < f.final_i[2, 4]


@pytest.mark.parametrize("family", ["old", "proposal", "final", "all", "none", "zero"])
def test_prefix_transpose_independent_field_roots(kernel, family):
    f = Fixture(); f.run(kernel); assert not f.error.item()
    refs = f.reference(); generator = torch.Generator().manual_seed(9)
    bars, flags, outputs, cotangents = [], [], [], []
    for name, states in zip(("old", "proposal", "final"), refs):
        bar = torch.full((len(states), f.width), float("nan"))
        connected = torch.zeros(len(states), 5, dtype=torch.bool)
        for row, state in enumerate(states):
            for field, ((offset, _, _, _), value) in enumerate(zip(f.fields.tolist(), state[3])):
                used = family in (name, "all", "zero") and (row+field) % 3 != 1
                if used:
                    connected[row, field] = True
                    gradient = torch.randn(value.shape, generator=generator) if family != "zero" else torch.zeros_like(value)
                    bar[row, offset:offset+value.numel()] = gradient.flatten()
                    outputs.append(value); cotangents.append(gradient)
        bars.append(bar); flags.append(connected)
    initial_bar = torch.empty_like(f.initial); initial_roots = torch.empty(f.owners, 5, dtype=torch.bool)
    write_bar = torch.empty_like(f.write); write_roots = torch.empty(f.rows, 5, dtype=torch.bool)
    args = (f.fields, f.write_i, f.old_i, f.proposal_i, f.mapping,
            bars[0], flags[0], bars[1], flags[1], bars[2], flags[2],
            initial_bar, initial_roots, write_bar, write_roots, f.error)
    kernel.tide_block_kv_sequence_reverse(*(x.data_ptr() for x in args), f.batch, f.length, f.nodes,
                                         f.width, f.write_width, f.capacity, 5)
    assert not f.error.item()
    leaves = [x for state in f.initial_fields+f.write_fields for x in state]
    expected = torch.autograd.grad(outputs, leaves, cotangents, allow_unused=True) if outputs else [None]*len(leaves)
    position = 0
    for kind, states, values, connected in (("initial", f.initial_fields, initial_bar, initial_roots),
                                           ("write", f.write_fields, write_bar, write_roots)):
        for row, state in enumerate(states):
            for field, ((off, wo, _, _), value) in enumerate(zip(f.fields.tolist(), state)):
                gradient = expected[position]; position += 1
                assert bool(connected[row, field]) == (gradient is not None), (kind, row, field)
                offset = off if kind == "initial" else wo
                actual = values[row, offset:offset+value.numel()].reshape(value.shape)
                torch.testing.assert_close(actual, gradient if gradient is not None else torch.zeros_like(value))


@pytest.mark.parametrize("fault,code", [("capacity", 76), ("clock", 75), ("overflow", 75), ("raw", 75)])
def test_sequence_capacity_clock_count_and_raw_refusals(kernel, fault, code):
    f = Fixture(horizon=0 if fault == "capacity" else 7)
    if fault == "clock":
        f.times[0, 0, 0] = f.initial_i[0, 0]
    elif fault == "overflow":
        f.initial_i[0, 1] = 2**63-1
    elif fault == "raw":
        f.initial[0, 18] = -1
    f.run(kernel)
    assert f.error.item() == code


def test_sequence_retains_raw_when_no_writes(kernel):
    f = Fixture(); f.write_i[:, :].zero_(); f.write_i[:, 0] = -1
    f.initial[1, 3:] = float("nan")  # Empty-state padding is not a logical field.
    f.run(kernel); assert not f.error.item()
    torch.testing.assert_close(f.final, f.initial, equal_nan=True)
    assert torch.equal(f.final_i, f.initial_i)
