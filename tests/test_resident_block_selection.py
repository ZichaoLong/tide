"""Block metadata and per-target gating; numerical composition is a device gate."""
import ctypes
from pathlib import Path
import subprocess

import pytest
import torch


@pytest.fixture(scope="module")
def kernel(tmp_path_factory):
    root = Path(__file__).resolve().parents[1]/"tools/device_online"
    work = tmp_path_factory.mktemp("block_selection")
    src, lib = work/"adapter.cpp", work/"adapter.so"
    src.write_text('#define TIDE_KERNEL_CPU\n'+''.join(
        f'#include "tide_{name}.cpp"\n' for name in
        ("block_selection_plan", "block_selection_roots", "block_selection_flags", "block_node_gate")))
    subprocess.run(["g++", "-std=c++17", "-shared", "-fPIC", "-O2", "-I"+str(root/"cuda"),
                    "-I"+str(root/"ascendc"), "-I"+str(Path(torch.__file__).parent/"include"),
                    str(src), "-o", str(lib)], check=True)
    result = ctypes.CDLL(str(lib))
    for name, pointers, integers in (("plan", 15, 4), ("roots", 6, 3), ("flags", 5, 3)):
        call = getattr(result, "tide_block_selection_"+name)
        call.argtypes = [ctypes.c_void_p]*pointers+[ctypes.c_int64]*integers
    result.tide_block_node_gate.argtypes = [ctypes.c_void_p]*6+[ctypes.c_int64]*4
    return result


def fixture():
    present = torch.tensor([[True, False, True], [False, True, True], [False, False, False]])
    times = torch.tensor([[2**55+7]*3, [2**55+12]*3, [-99]*3])
    source = torch.tensor([[0, -99, 2], [-99, 1, 2], [-99]*3])
    offsets = torch.tensor([0, 2, 3]); members = torch.tensor([0, 2, 1])
    # Region 0 has nodes 0/2, region 1 has node 1; node index is not region rank.
    fields = [present, times, source, offsets, members,
              torch.empty(9, dtype=torch.int64), torch.empty(9, dtype=torch.bool),
              torch.empty(9, dtype=torch.int64), torch.empty(6, 2, dtype=torch.int64),
              torch.empty(9, dtype=torch.int64), torch.empty(6, 2, dtype=torch.int64),
              torch.empty(6, dtype=torch.int64), torch.empty(6, dtype=torch.int64),
              torch.empty(9, dtype=torch.int64), torch.zeros(1, dtype=torch.int32)]
    return fields


def test_complete_frames_and_independent_members(kernel):
    f = fixture(); kernel.tide_block_selection_plan(*(x.data_ptr() for x in f), 3, 3, 2, 2)
    assert not f[-1].item()
    expected_order, expected_ids, expected_lengths = [], [], []
    for frame in range(3):
        for nodes in ([0, 2], [1]):
            found = [v for v in nodes if f[0][frame, v]]
            expected_order.append([frame*3+v for v in found]+[9]*(2-len(found)))
            expected_ids.append(found+[0]*(2-len(found))); expected_lengths.append(len(found))
    assert f[8].tolist() == expected_order
    assert f[10].tolist() == expected_ids
    assert f[12].tolist() == expected_lengths
    assert f[11].tolist() == [2**55+7, 0, 2**55+12, 2**55+12, 0, 0]
    assert f[5].tolist() == [0, -1, 2, -1, 1, 2, -1, -1, -1]
    for row in range(9):
        if f[0].flatten()[row]:
            assert f[8].flatten()[f[9][row]] == row
        else:
            assert f[9][row] == 12


def test_region_time_mismatch_and_invalid_owners_fail(kernel):
    for kind in ("time", "owner", "duplicate"):
        f = fixture()
        if kind == "time": f[1][0, 2] += 1
        elif kind == "owner": f[2][0, 2] = -1
        else: f[4][2] = 0
        kernel.tide_block_selection_plan(*(x.data_ptr() for x in f), 3, 3, 2, 2)
        assert f[-1].item() == 70


def test_control_roots_ignore_absence_but_preserve_connected_zero(kernel):
    f = fixture(); kernel.tide_block_selection_plan(*(x.data_ptr() for x in f), 3, 3, 2, 2)
    gradient = torch.full((9,), float('nan')); gradient[0] = 0; gradient[5] = .7
    roots = torch.tensor([True, True, False, False, False, True, True, True, True])
    packed = torch.empty(6, 2); flags = torch.empty(6, dtype=torch.bool)
    kernel.tide_block_selection_roots(*(x.data_ptr() for x in (f[8], gradient, roots, packed, flags, f[-1])), 6, 2, 9)
    assert not f[-1].item()
    assert flags.tolist() == [True, False, True, False, False, False]
    assert packed[0].eq(0).all() and packed[2, 0] == .7
    assert packed.isfinite().all()
    region_flags = torch.ones(6, 6, dtype=torch.bool); unpacked = torch.empty(9, 6, dtype=torch.bool)
    kernel.tide_block_selection_flags(*(x.data_ptr() for x in (f[6], f[13], region_flags, unpacked, f[-1])), 0, 9, 6)
    assert torch.equal(unpacked, f[0].flatten()[:, None].expand(9, 6))


def test_per_target_gate_does_not_broadcast_region_activity(kernel):
    tasks = torch.tensor([[2, 0, i, 0, 0] for i in range(3)])
    selected = torch.tensor([0, 1, 2, 3]); meta = torch.tensor([3, 2, 7, 0])
    active = torch.tensor([[False, True, False], [False, False, True]])
    emitted = torch.ones(4, dtype=torch.bool); error = torch.zeros(1, dtype=torch.int32)
    kernel.tide_block_node_gate(*(x.data_ptr() for x in (tasks, selected, meta, active, emitted, error)), 3, 4, 2, 3)
    assert not error.item() and emitted.tolist() == [False, True, True, False]


def test_node_gate_rejects_wrong_numerical_phase(kernel):
    tasks = torch.tensor([[1, 0, 0, 0, 0]])
    selected = torch.tensor([0]); meta = torch.tensor([1, 1, 7, 0])
    active = torch.ones(1, 1, dtype=torch.bool)
    emitted = torch.zeros(1, dtype=torch.bool); error = torch.zeros(1, dtype=torch.int32)
    kernel.tide_block_node_gate(*(x.data_ptr() for x in (tasks, selected, meta, active, emitted, error)), 1, 1, 1, 1)
    assert error.item() == 70 and not emitted.item()


def test_metadata_lowering_preserves_physical_edges(tmp_path):
    root = Path(__file__).resolve().parents[1]/"tools/device_online"
    source, executable = tmp_path/"plan.cpp", tmp_path/"plan"
    source.write_text(r'''
#include "settle_block_plan.h"
#include <cassert>
using namespace tide::device_online;
int main(){
  auto p=make_settle_block_plan({0,1,2,0},{0,0,7},{{0,2,7},{0,2,7},{1,2,7}},
                              {0,0,1,3},(int64_t{1}<<54)+1,3,8,1<<20);
  assert(p.dispatch.tasks.size()==15&&p.dispatch.dependencies.size()==15);
  assert(p.input_tasks[0]==p.input_tasks[1]);
  assert(p.edge_dependencies[0]!=p.edge_dependencies[1]);
  assert(p.dispatch.dependencies[p.edge_dependencies[0]]==p.dispatch.dependencies[p.edge_dependencies[1]]);
  assert(p.dispatch.tasks[p.local[0]][3]==((int64_t{1}<<54)+1)*3);
  assert((p.member_offsets==std::vector<int64_t>{0,2,3,4}));
  assert((p.members==std::vector<int64_t>{0,3,1,2}));
  assert(p.dispatch.dependencies[2][0]==p.gate[0]);
  bool rejected=false;try{make_settle_block_plan({0,1},{0,0},{{0,1,1}},{0},0,1,8,1<<20);}
  catch(const std::invalid_argument&){rejected=true;}assert(rejected);
  rejected=false;try{make_settle_block_plan({0},{0},{},{0},0,1,8,1);}
  catch(const std::invalid_argument&){rejected=true;}assert(rejected);
}''')
    subprocess.run(["g++", "-std=c++17", "-O2", "-I"+str(root), str(source), "-o", str(executable)], check=True)
    subprocess.run([str(executable)], check=True)
