"""Physical routing layout; CPU semantic-source checks, separate from NPU numerics."""
import ctypes
from pathlib import Path
import subprocess

import pytest
import torch


@pytest.fixture(scope="module")
def kernels(tmp_path_factory):
    root = Path(__file__).resolve().parents[1]/"tools/device_online"
    work = tmp_path_factory.mktemp("block_flow")
    src, lib = work/"adapter.cpp", work/"adapter.so"
    src.write_text('#define TIDE_KERNEL_CPU\n'+''.join(f'#include "{name}.cpp"\n' for name in
                   ("tide_block_flow_fibers", "tide_block_flow_publish", "tide_block_raw_roots",
                    "tide_block_flow_reverse_ports", "tide_block_flow_reverse_gather",
                    "tide_block_flow_reverse_fibers", "tide_block_flow_reverse_choose", "tide_block_flow_boundary")))
    subprocess.run(["g++", "-std=c++17", "-shared", "-fPIC", "-O2", "-I"+str(root/"cuda"),
                    "-I"+str(root/"ascendc"), "-I"+str(Path(torch.__file__).parent/"include"), str(src), "-o", str(lib)], check=True)
    dll = ctypes.CDLL(str(lib))
    for name, pointers, integers in (("fibers", 15, 9), ("publish", 9, 5)):
        getattr(dll, "tide_block_flow_"+name).argtypes = [ctypes.c_void_p]*pointers+[ctypes.c_int64]*integers
    dll.tide_block_raw_roots.argtypes = [ctypes.c_void_p]*8+[ctypes.c_int64]*5
    for name, pointers, integers in (("ports", 9, 4), ("gather", 13, 4), ("fibers", 9, 5), ("choose", 5, 2)):
        getattr(dll, "tide_block_flow_reverse_"+name).argtypes = [ctypes.c_void_p]*pointers+[ctypes.c_int64]*integers
    dll.tide_block_flow_boundary.argtypes = [ctypes.c_void_p]*8+[ctypes.c_int64]*8
    return dll


def fibers(kernels, bad_time=False):
    b, l, n, a, d, inputs, nodes = 2, 2, 2, 3, 3, 2, 3
    atoms = torch.tensor([[[1, 0, 1], [0, 1, 1], [1, 0, 1]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
    counts, bank = torch.tensor([3, 1]), torch.tensor([0, 1])
    positions = torch.tensor([[2**54+17, 2**54+20], [2**54+30, 2**54+32]])
    if bad_time:
        positions[0, 1] = positions[0, 0]
    ext = torch.arange(b*l*inputs*d, dtype=torch.float32).reshape(b*l, inputs, d)
    ext_present = torch.tensor([[True, False], [True, True], [False, True], [False, False]])
    messages = torch.arange(b*l*nodes*d, dtype=torch.float32).reshape(b*l, nodes, d)/10
    messages[0, 0] = torch.tensor([.5, 0, 0])
    live = torch.tensor([[True, False, False], [False, False, True], [True, True, True], [False, False, False]])
    payload = torch.empty(b*l*n, a, d); lengths = torch.empty(b*l*n, dtype=torch.int64)
    owners = torch.empty(b*l*n, a, dtype=torch.int64); node_owners = torch.empty(b*l*n, dtype=torch.int64)
    times = torch.empty(b*l*n, dtype=torch.int64); mapping = torch.empty(b*l*n, a, 3, dtype=torch.int64)
    error = torch.zeros(1, dtype=torch.int32)
    arrays = atoms, counts, bank, positions, ext, ext_present, messages, live, payload, lengths, owners, node_owners, times, mapping, error
    kernels.tide_block_flow_fibers(*(x.data_ptr() for x in arrays), b, l, n, a, d, inputs, nodes, 2, 4)
    return arrays


def test_physical_fibers_keep_same_class_duplicates_and_zero_G(kernels):
    atoms, counts, bank, positions, ext, ep, message, mp, payload, lengths, owners, no, times, mapping, error = fibers(kernels)
    assert not error.item()
    for frame in range(4):
        for node in range(2):
            expected, indices = [], []
            for slot in range(counts[node]):
                kind, source, owner = atoms[node, slot].tolist()
                if (mp if kind else ep)[frame, source]:
                    expected.append((message if kind else ext)[frame, source])
                    indices.append([kind, frame*(3 if kind else 2)+source, slot])
            row = frame*2+node
            assert lengths[row] == len(expected)
            if expected:
                assert torch.equal(payload[row, :len(expected)], torch.stack(expected))
                assert mapping[row, :len(expected)].tolist() == indices
            assert payload[row, len(expected):].eq(0).all()
            assert times[row] == positions.flatten()[frame]*4+2
    assert lengths[0] == 2
    assert payload[0, :2, 1:].eq(0).all()  # Present zero content is retained twice.
    assert mapping[0, :2, 2].tolist() == [0, 2]


def test_nonincreasing_sequence_positions_refused(kernels):
    assert fibers(kernels, True)[-1].item() == 79


def test_physical_ports_and_guarded_publication(kernels):
    nodes = torch.tensor([0, 0, 2]); delay = torch.tensor([2, 4, 1])
    values = torch.tensor([[[.3, 0., 0.], [.2, 1., 2.], [0., 0., 0.]]])
    present = torch.tensor([[True, True, False]])
    times = torch.tensor([[2**55+3, 2**55+4, 0]])
    output = torch.full((1, 3, 3), -7.); live = torch.ones(1, 3, dtype=torch.bool)
    clock = torch.full((1, 3), -7, dtype=torch.int64); error = torch.tensor([1], dtype=torch.int32)
    args = nodes, delay, values, present, times, output, live, clock, error
    kernels.tide_block_flow_publish(*(x.data_ptr() for x in args), 2, 1, 3, 3, 3)
    assert output.eq(-7).all() and clock.eq(-7).all()
    error.zero_(); kernels.tide_block_flow_publish(*(x.data_ptr() for x in args), 2, 1, 3, 3, 3)
    assert live.tolist() == [[True, True, False]]
    assert torch.equal(output[0, 0], output[0, 1])
    assert clock.tolist() == [[2**55+5, 2**55+7, 0]]


def test_raw_roots_fold_batch_time_and_preserve_connected_zero(kernels):
    fields = torch.tensor([[0, 0, 2, 0], [2, 2, 1, 1]])
    metadata = torch.tensor([[-1, 0, 0, 0, 0, 0], [5, 2, 2, 0, 1, 5], [3, 1, 1, 0, 3, 0]])
    mapping = torch.tensor([0, 0, 1])
    bar = torch.tensor([[0., 0., float("nan"), float("nan")], [1., 2., 3., 4.], [float("nan"), float("nan"), 0., float("nan")]])
    roots = torch.tensor([[True, True], [True, True], [False, True]])
    output = torch.zeros(2, 4); connected = torch.zeros(2, 2, dtype=torch.bool); error = torch.zeros(1, dtype=torch.int32)
    args = fields, metadata, mapping, bar, roots, output, connected, error
    kernels.tide_block_raw_roots(*(x.data_ptr() for x in args), 3, 2, 4, 2, 2)
    assert not error.item()
    assert output.tolist() == [[1., 2., 3., 4.], [0., 0., 0., 0.]]
    assert connected.tolist() == [[True, True], [False, True]]


def test_duplicate_ports_have_independent_strength_and_content_roots(kernels):
    sources = torch.tensor([1, 1, 0])
    present = torch.tensor([[True, True, False], [False, True, True]])
    mass = torch.tensor([[.2, .3, float("nan")], [float("nan"), 0., 0.]])
    mr = torch.tensor([[True, True, True], [False, True, False]])
    content = torch.tensor([[[float("nan"), float("nan")], [.5, .6], [float("nan"), float("nan")]],
                            [[float("nan"), float("nan")], [0., 0.], [.7, .8]]])
    cr = torch.tensor([[False, True, True], [True, True, True]])
    out = torch.zeros(2, 2, 3); flags = torch.zeros(2, 2, 2, dtype=torch.bool)
    error = torch.zeros(1, dtype=torch.int32)
    arrays = sources, present, mass, mr, content, cr, out, flags, error
    kernels.tide_block_flow_reverse_ports(*(x.data_ptr() for x in arrays), 2, 3, 2, 3)
    assert not error.item()
    assert torch.equal(out, torch.tensor([[[0., 0., 0.], [.5, .5, .6]], [[0., .7, .8], [0., 0., 0.]]]))
    assert flags.tolist() == [[[False, False], [True, True]], [[False, True], [True, True]]]


def test_actual_fiber_transpose_sums_each_physical_use(kernels):
    mapping = torch.tensor([[[1, 0, 0], [1, 0, 2], [0, 1, 1]], [[0, 1, 0], [-1, -1, -1], [-1, -1, -1]]])
    lengths = torch.tensor([3, 1])
    grad = torch.tensor([[[.1, .2, .3], [.4, .5, .6], [0., 0., 0.]], [[.2, .3, .4], [float("nan")]*3, [float("nan")]*3]])
    roots = torch.tensor([[True, True, True], [True, True, True]])
    ext = torch.zeros(2, 3); er = torch.zeros(2, dtype=torch.bool)
    msg = torch.zeros(2, 3); mr = torch.zeros(2, 2, dtype=torch.bool)
    error = torch.zeros(1, dtype=torch.int32)
    arrays = mapping, lengths, grad, roots, ext, er, msg, mr, error
    kernels.tide_block_flow_reverse_fibers(*(x.data_ptr() for x in arrays), 2, 3, 3, 2, 2)
    assert not error.item()
    torch.testing.assert_close(msg, torch.tensor([[.5, .7, .9], [0., 0., 0.]]))
    torch.testing.assert_close(ext, torch.tensor([[0., 0., 0.], [.2, .3, .4]]))
    assert er.tolist() == [False, True] and mr.tolist() == [[True, True], [False, False]]


def test_reverse_uses_actual_forward_journal_and_refuses_incomplete(kernels):
    journal = torch.tensor([[1, 1, 0], [0, 0, 1], [2, 1, 2]])
    ledger = torch.tensor([2, 3, 3, 0]); cursor = torch.zeros(1, dtype=torch.int64)
    branch = torch.zeros(1, dtype=torch.int32); error = torch.zeros(1, dtype=torch.int32)
    arrays = journal, ledger, cursor, branch, error
    call = lambda command: kernels.tide_block_flow_reverse_choose(*(x.data_ptr() for x in arrays), 3, command)
    call(0); observed = []
    for _ in range(4):
        call(1); observed.append(branch.item())
    assert observed == [3, 1, 2, 0] and not error.item()
    ledger[2] = 2; call(0)
    assert error.item() == 80 and branch.item() == 0


def test_reverse_gather_combines_direct_and_downstream_without_false_roots(kernels):
    nodes = torch.tensor([2, 0]); present = torch.tensor([[True, False]])
    incoming = torch.tensor([[[float("nan")]*3, [float("nan")]*3, [.2, .3, .4]]])
    flags = torch.tensor([[[True, True], [False, False], [True, True]]])
    dm = torch.tensor([[.1, float("nan")]]); dmr = torch.tensor([[True, True]])
    dc = torch.full((1, 2, 2), float("nan")); dcr = torch.zeros(1, 2, dtype=torch.bool)
    mass = torch.empty_like(dm); mr = torch.empty_like(dmr)
    content = torch.empty_like(dc); cr = torch.empty_like(dcr); error = torch.zeros(1, dtype=torch.int32)
    arrays = nodes, present, incoming, flags, dm, dmr, dc, dcr, mass, mr, content, cr, error
    kernels.tide_block_flow_reverse_gather(*(x.data_ptr() for x in arrays), 1, 2, 3, 3)
    assert not error.item()
    torch.testing.assert_close(mass, torch.tensor([[.3, 0.]]))
    assert torch.equal(content, torch.tensor([[[.3, .4], [0., 0.]]]))
    assert mr.tolist() == cr.tolist() == [[True, False]]


def test_complete_cut_metadata_uses_actual_candidates_and_physical_input_counts(kernels):
    nodes = torch.tensor([0, 2]); present = torch.tensor([[[True, False], [False, True]], [[False, False], [True, True]]])
    times = torch.tensor([[[2**55, -99], [-99, 2**55+6]], [[-99, -99], [2**55+26, 2**55+26]]])
    seen = torch.zeros(2, 3, dtype=torch.bool); history = torch.full((2, 2), -1, dtype=torch.int64)
    count = torch.full((2, 2), -1, dtype=torch.int64); clocks = count.clone(); error = torch.zeros(1, dtype=torch.int32)
    arrays = nodes, present, times, seen, history, count, clocks, error
    kernels.tide_block_flow_boundary(*(x.data_ptr() for x in arrays), 1, 2, 2, 2, 3, 1, 2, 2)
    assert not error.item() and seen.tolist() == [[True, False, True], [True, False, True]]
    assert history.tolist() == [[-1, 2**55+6], [-1, 2**55+26]]
    ranks = torch.tensor([0, 1]); positions = torch.tensor([[2**54, 2**54+3], [2**54+10, 2**54+13]])
    arrays = ranks, present, positions, seen, history, count, clocks, error
    kernels.tide_block_flow_boundary(*(x.data_ptr() for x in arrays), 0, 2, 2, 2, 3, 0, 2, 2)
    assert not error.item() and count.eq(0).all()  # One actual occurrence per physical port.
    assert clocks.tolist() == [[2**55, 2**55+7], [2**55+26, 2**55+27]]
    kernels.tide_block_flow_boundary(*(x.data_ptr() for x in arrays), 0, 2, 2, 2, 3, 0, 2, 2)
    assert error.item() == 81  # Stale input occurrence cannot silently advance.
