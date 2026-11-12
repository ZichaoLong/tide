"""Actual device partition/copy sources + CPU GEMMs versus scalar FP64 formulas.

This does not execute DeviceProgram/CANN branches; device integration has its
own weighted_projection/head/kv_update checks and capacity witness.
"""
import ctypes
from pathlib import Path
import subprocess
import sys
import pytest
import torch


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    device = Path(__file__).resolve().parents[1]/"tools/device_online"
    directory = tmp_path_factory.mktemp("bank-gemm")
    generated = directory/"generated"
    subprocess.run([sys.executable, str(device/"cuda/generate_kernels.py"), "--source", str(device/"ascendc"),
                    "--output", str(generated), "--backend", "cpu"], check=True)
    names = ("plan", "rows", "snapshot", "next")
    path = directory/"bank.so"
    subprocess.run(["g++", "-std=c++17", "-shared", "-fPIC", "-O2", "-ffp-contract=off",
                    "-I"+str(device/"cuda"), "-I"+str(device/"ascendc"),
                    "-I"+str(Path(torch.__file__).parent/"include"),
                    *(str(generated/f"tide_weighted_bank_{n}.cpp") for n in names), "-o", str(path)], check=True)
    lib = ctypes.CDLL(str(path))
    for name, pointers, integers in zip(names, (8, 6, 6, 4), (3, 6, 5, 4)):
        f = getattr(lib, "tide_cuda_launch_tide_weighted_bank_"+name)
        f.argtypes = [ctypes.c_uint32, ctypes.c_void_p]+[ctypes.c_void_p]*pointers+[ctypes.c_int64]*integers
    binary = directory/"layout-check"
    subprocess.run(["g++", "-std=c++17", "-O2", str(device/"weighted_bank_layout_check.cpp"), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
    return lib


def launch(lib, name, tensors, *values, blocks=1):
    f = getattr(lib, "tide_cuda_launch_tide_weighted_bank_"+name)
    assert f(blocks, None, *(x.data_ptr() for x in tensors), *values) == 0


def plan(lib, owners, banks, roots=None):
    n = owners.numel()
    roots_arg = torch.zeros(n, dtype=torch.bool) if roots is None else roots
    permutation = torch.full_like(owners, -99)
    offsets = torch.full((banks+1,), -99, dtype=torch.int64)
    cursors = torch.zeros(banks, dtype=torch.int64)
    connected = torch.ones(n, dtype=torch.bool)
    used = torch.ones(banks, dtype=torch.bool)
    error = torch.zeros(1, dtype=torch.int32)
    launch(lib, "plan", (owners, roots_arg, permutation, offsets, cursors, connected, used, error), n, banks, roots is not None)
    return dict(perm=permutation, offsets=offsets, connected=connected, used=used, error=error)


def groups(lib, p, rows):
    tile = min(rows, 64)
    cursor = torch.full((2,), -99, dtype=torch.int64)
    branch = torch.full((1,), -99, dtype=torch.int32)
    args = (p["offsets"], cursor, branch, p["error"])
    launch(lib, "next", args, rows, len(p["used"]), tile, True)
    for _ in range(rows+1):
        launch(lib, "next", args, rows, len(p["used"]), tile, False)
        assert p["error"].item() == 0
        if not branch.item(): return
        assert branch.item() == 1
        yield cursor[0].item(), cursor[1].item(), tile
    raise AssertionError("device bank/tile loop failed to terminate")


def transfer(lib, p, source, destination, bank, cap, *, start=0, scatter=False, code=20):
    cursor = torch.tensor([bank, start], dtype=torch.int64)
    launch(lib, "rows", (p["perm"], p["offsets"], cursor, source, destination, p["error"]),
           p["perm"].numel(), len(p["used"]), cap, source.shape[-1], scatter, code, blocks=4)


def snapshot(lib, p, matrix, bias=None):
    saved = torch.full_like(matrix, 19)
    bias = torch.empty(0) if bias is None else bias
    saved_bias = torch.full_like(bias, 19)
    launch(lib, "snapshot", (p["used"], matrix, bias, saved, saved_bias, p["error"]),
           *matrix.shape[:1], matrix.shape[-1], matrix.shape[-2], bool(bias.numel()), 20, blocks=4)
    return saved, saved_bias


def forward(lib, x, owners, matrix, bias=None, epsilon=None):
    p = plan(lib, owners, len(matrix)); w, b = snapshot(lib, p, matrix, bias)
    clean = torch.where((owners >= 0)[:, None], x, torch.zeros_like(x))
    rstd = (clean.square().mean(-1, keepdim=True)+epsilon).rsqrt() if epsilon else torch.ones(len(x), 1)
    normalized = clean*rstd
    output = torch.zeros(len(x), w.shape[1])
    for owner, start, cap in groups(lib, p, len(x)):
        packed = torch.empty(cap, x.shape[1]); transfer(lib, p, normalized, packed, owner, cap, start=start)
        y = packed@w[owner].T
        if b.numel(): y = y+b[owner]
        transfer(lib, p, y, output, owner, cap, start=start, scatter=True)
    assert p["error"].item() == 0
    return output, (normalized, owners.clone(), w, b, rstd, epsilon)


def backward(lib, tape, bar, roots):
    x, owners, w, b, rstd, epsilon = tape
    p = plan(lib, owners, len(w), roots)
    dx, dw, db = torch.zeros_like(x), torch.zeros_like(w), torch.zeros_like(b)
    for owner, start, cap in groups(lib, p, len(x)):
        px, pg = torch.empty(cap, x.shape[1]), torch.empty(cap, bar.shape[1])
        transfer(lib, p, x, px, owner, cap, start=start, code=23); transfer(lib, p, bar, pg, owner, cap, start=start, code=23)
        dw[owner] += pg.T@px
        if b.numel(): db[owner] += pg.sum(0)
        transfer(lib, p, pg@w[owner], dx, owner, cap, start=start, scatter=True, code=23)
    if epsilon: dx = rstd*(dx-x*(dx*x).mean(-1, keepdim=True))
    assert p["error"].item() == 0
    return dx, dw, db, p


def formula(x, owners, w, b, roots, bar, epsilon):
    x, w = x.double().requires_grad_(), w.double().requires_grad_()
    b = None if b is None else b.double().requires_grad_()
    result, loss = [], x.new_zeros(())
    for row, owner in enumerate(owners.tolist()):
        if owner < 0:
            result.append(x.new_zeros(w.shape[1])); continue
        u = x[row]/(epsilon+x[row].square().mean()).sqrt() if epsilon else x[row]
        y = torch.mv(w[owner], u)
        if b is not None: y = y+b[owner]
        result.append(y)
        if roots[row]: loss = loss+(y*bar[row].double()).sum()
    leaves = (x, w) if b is None else (x, w, b)
    grads = torch.autograd.grad(loss, leaves, allow_unused=True) if loss.requires_grad else (None,)*len(leaves)
    return torch.stack(result), tuple(torch.zeros_like(x) if g is None else g for x, g in zip(leaves, grads))


@pytest.mark.parametrize("epsilon", [None, .13])
@pytest.mark.parametrize("bias", [False, True])
def test_formula_vjp_shared_owners_private_banks_and_saved_versions(library, epsilon, bias):
    g = torch.Generator().manual_seed(82)
    x = torch.randn(17, 7, generator=g)*.2
    owners = torch.tensor([2, 0, -1, 2, 1, 0, 1, 2, 0, 2, -1, 1, 2, 0, 2, 2, 2])
    w = torch.randn(4, 5, 7, generator=g)*.2; w[1].copy_(w[0])  # equal values are independent banks
    b = torch.randn(4, 5, generator=g)*.2 if bias else None
    roots = torch.tensor([1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1], dtype=torch.bool)
    bar = torch.randn(17, 5, generator=g); bar[0].zero_()
    x[owners < 0] = float("nan"); bar[~roots | (owners < 0)] = float("nan")
    w[3] = float("nan")
    if b is not None: b[3] = float("nan")
    expected, grads = formula(x, owners, w, b, roots, bar, epsilon)
    y, tape = forward(library, x, owners, w, b, epsilon)
    x.fill_(float("nan")); w.fill_(float("nan")); owners.fill_(-99)
    if b is not None: b.fill_(float("nan"))
    dx, dw, db, p = backward(library, tape, bar, roots)
    for actual, ref in zip((y, dx, dw, db), (expected, *grads)):
        torch.testing.assert_close(actual.double(), ref, atol=1e-6, rtol=1e-5)
    assert p["connected"][0] and p["used"].tolist() == [True, True, True, False]
    assert dw[3].eq(0).all() and not torch.allclose(dw[0], dw[1])


def test_actual_alias_contributions_across_owner_slots(library):
    x = torch.arange(30, dtype=torch.float32).reshape(10, 3)*.03
    shared = torch.arange(12, dtype=torch.float64).reshape(4, 3).requires_grad_()
    owners = torch.tensor([0, 1, 0, 1, 1, 0, 1, 1, 0, 1]); bar = torch.cos(torch.arange(40).reshape(10, 4).float())
    y, tape = forward(library, x, owners, torch.stack([shared.detach(), shared.detach()]).float())
    _, dw, _, _ = backward(library, tape, bar, torch.ones(10, dtype=torch.bool))
    expected = torch.autograd.grad(sum((shared@x[r].double()*bar[r].double()).sum() for r in range(10)), shared)[0]
    torch.testing.assert_close(dw.sum(0).double(), expected, atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(y.double(), x.double()@shared.T, atol=2e-6, rtol=1e-5)


def test_empty_groups_and_disconnected_poison(library):
    owners = torch.full((7,), -1); matrix = torch.full((3, 5, 4), float("nan"))
    y, tape = forward(library, torch.full((7, 4), float("nan")), owners, matrix)
    dx, dw, _, p = backward(library, tape, torch.full_like(y, float("nan")), torch.ones(7, dtype=torch.bool))
    assert y.eq(0).all() and dx.eq(0).all() and dw.eq(0).all()
    assert not p["used"].any() and not p["connected"].any()
    owners.zero_(); matrix.zero_(); y, tape = forward(library, torch.ones(7, 4), owners, matrix)
    dx, dw, _, p = backward(library, tape, torch.full_like(y, float("nan")), torch.zeros(7, dtype=torch.bool))
    assert dx.eq(0).all() and dw.eq(0).all() and not p["used"].any()


@pytest.mark.parametrize("rows", [1, 3, 6, 17, 64, 65, 129, 257])
def test_stable_partition_tiles_and_capacity_refusal(library, rows):
    owners = torch.arange(rows).remainder(4)-1; p = plan(library, owners, 4)
    expected = [r for owner in range(4) for r, value in enumerate(owners.tolist()) if value == owner]
    assert p["error"].item() == 0 and p["perm"][:len(expected)].tolist() == expected
    tiles = list(groups(library, p, rows)); capacity = min(rows, 64)
    expected_tiles = [(b, start, capacity) for b in range(4) for start in range(0, int((owners == b).sum()), capacity)]
    assert tiles == expected_tiles
    assert sum(t[2] for t in tiles) <= len(expected)+int(p["used"].sum())*(capacity-1)
    bad = owners.clone(); bad[0] = 4; rejected = plan(library, bad, 4)
    assert rejected["error"].item() == 2 and not rejected["used"].any()
    full = plan(library, torch.zeros(3, dtype=torch.int64), 1)
    transfer(library, full, torch.ones(3, 2), torch.empty(2, 2), 0, 2, start=4)
    assert full["error"].item() == 2


def test_device_bank_cursor_replay_invalid_and_sticky_error(library):
    offsets = torch.tensor([0, 0, 129, 129, 130, 195], dtype=torch.int64)
    cursor = torch.full((2,), -99, dtype=torch.int64)
    branch = torch.ones(1, dtype=torch.int32); error = torch.zeros(1, dtype=torch.int32)
    args = (offsets, cursor, branch, error)
    for _ in range(2):
        launch(library, "next", args, 195, 5, 64, True)
        actual = []
        for _ in range(8):
            launch(library, "next", args, 195, 5, 64, False)
            actual.append((*cursor.tolist(), branch.item()))
        assert actual == [(1, 0, 1), (1, 64, 1), (1, 128, 1), (3, 0, 1), (4, 0, 1), (4, 64, 1), (5, 0, 0), (5, 0, 0)]
        assert not error.item()
    cursor.fill_(-2)
    launch(library, "next", args, 195, 5, 64, False)
    assert error.item() == 2 and branch.item() == 0
    cursor.zero_(); error.fill_(23); branch.fill_(99)
    launch(library, "next", args, 195, 5, 64, False)
    assert error.item() == 23 and branch.item() == 0 and cursor.tolist() == [0, 0]


def test_tile_cursor_keeps_int64_counts_and_rejects_bad_offsets(library):
    base = 1 << 54
    offsets = torch.tensor([0, base+65], dtype=torch.int64)
    cursor = torch.tensor([0, base], dtype=torch.int64)
    branch = torch.zeros(1, dtype=torch.int32); error = torch.zeros_like(branch)
    args = (offsets, cursor, branch, error)
    launch(library, "next", args, base+65, 1, 64, False)
    assert cursor.tolist() == [0, base+64] and branch.item() == 1 and not error.item()
    launch(library, "next", args, base+65, 1, 64, False)
    assert cursor.tolist() == [1, 0] and branch.item() == 0 and not error.item()
    for broken in ([0, -1], [0, 130], [-(1 << 63), 129]):
        offsets.copy_(torch.tensor(broken)); cursor.copy_(torch.tensor([-1, 0])); error.zero_()
        launch(library, "next", args, 129, 1, 64, False)
        assert error.item() == 2 and branch.item() == 0


def test_used_nonfinite_snapshot_and_sticky_failure(library):
    p = plan(library, torch.tensor([0, -1]), 2)
    matrix = torch.ones(2, 3, 4); matrix[0, 0, 0] = float("inf")
    snapshot(library, p, matrix)
    assert p["error"].item() == 20
    out = torch.full((2, 3), 17.)
    transfer(library, p, torch.ones(2, 3), out, 0, 1)
    assert out.eq(17).all() and p["error"].item() == 20


@pytest.mark.parametrize("bias,epsilon", [(False, None), (True, .13)])
def test_multiple_tiles_odd_tails_independent_roots_and_saved_banks(library, bias, epsilon):
    gen = torch.Generator().manual_seed(81)
    owners = torch.tensor([2]*129+[0]*65+[4]*3+[-1]*17+[3]*70)
    owners = owners[torch.randperm(len(owners), generator=gen)]
    x = torch.randn(len(owners), 7, generator=gen)*.2
    w = torch.randn(6, 5, 7, generator=gen)*.2; w[2].copy_(w[0])
    b = torch.randn(6, 5, generator=gen)*.2 if bias else None
    roots = owners != 3; bar = torch.randn(len(owners), 5, generator=gen)
    bar[owners == 4] = 0  # Present, connected-zero bank; not an empty group.
    bar[(owners < 0) | ~roots] = float("nan"); x[owners < 0] = float("nan")
    w[[1, 5]] = float("nan")
    if b is not None: b[[1, 5]] = float("nan")
    expected, grads = formula(x, owners, w, b, roots, bar, epsilon)
    y, tape = forward(library, x, owners, w, b, epsilon)
    x.fill_(float("nan")); w.fill_(float("nan")); owners.fill_(-99)
    if b is not None: b.fill_(float("nan"))
    first = backward(library, tape, bar, roots); second = backward(library, tape, bar, roots)
    for actual, ref in zip((y, *first[:3]), (expected, *grads)):
        torch.testing.assert_close(actual.double(), ref, atol=2e-6, rtol=1e-5)
    for a, b in zip(first[:3], second[:3]): assert torch.equal(a, b)
    assert first[3]["used"].tolist() == [True, False, True, False, True, False]
    assert first[1][4].eq(0).all() and first[3]["connected"][tape[1] == 4].all()
    tiles = list(groups(library, first[3], len(tape[1])))
    assert [(b, start) for b, start, _ in tiles] == [(0, 0), (0, 64), (2, 0), (2, 64), (2, 128), (4, 0)]


def test_tile_tail_does_not_read_or_publish_padding_and_guards_later_tiles(library):
    p = plan(library, torch.zeros(129, dtype=torch.int64), 2)
    source = torch.arange(129*3, dtype=torch.float32).reshape(129, 3)
    packed = torch.full((64, 3), float("nan"))
    transfer(library, p, source, packed, 0, 64, start=128)
    assert torch.equal(packed[0], source[-1]) and packed[1:].eq(0).all()
    packed[1:] = float("nan"); out = torch.full_like(source, 19)
    transfer(library, p, packed, out, 0, 64, start=128, scatter=True)
    assert not p["error"].item() and out[:-1].eq(19).all() and torch.equal(out[-1], source[-1])
    source[70, 1] = float("nan")
    transfer(library, p, source, packed, 0, 64, start=64, code=23)
    assert p["error"].item() == 23
    for bank, start in [(0, 129), (0, 1), (1, 0), (2, 0), (-1, 0)]:
        p["error"].zero_(); packed.fill_(19)
        transfer(library, p, source, packed, bank, 64, start=start)
        assert p["error"].item() == 2 and packed.eq(19).all()
    p["error"].zero_(); p["offsets"][1] = 130
    transfer(library, p, source, packed, 0, 64)
    assert p["error"].item() == 2


def test_two_affine_head_transpose_and_tied_parameter_roles(library):
    generator = torch.Generator().manual_seed(17)
    x = torch.randn(9, 3, generator=generator)*.2
    owners = torch.tensor([0, 1, 0, -1, 0, 1, 0, 1, 0])
    w = torch.randn(3, 3, 3, generator=generator)*.2
    b = torch.randn(3, 3, generator=generator)*.2
    pre, first = forward(library, x, owners, w, b)
    y, second = forward(library, torch.nn.functional.silu(pre), owners, w, b)
    bar = torch.randn(y.shape, generator=generator); bar[0].zero_()
    roots = torch.tensor([1, 0, 1, 1, 0, 1, 1, 1, 0], dtype=torch.bool)
    dh, dw2, db2, p = backward(library, second, bar, roots)
    dz = torch.ops.aten.silu_backward(dh, pre)
    dx, dw1, db1, _ = backward(library, first, dz, p["connected"])
    xx, ww, bb = [v.double().requires_grad_() for v in (x, w, b)]
    outputs, loss = [], xx.new_zeros(())
    for row, owner in enumerate(owners.tolist()):
        v = torch.mv(ww[owner], torch.nn.functional.silu(torch.mv(ww[owner], xx[row])+bb[owner]))+bb[owner] if owner >= 0 else xx.new_zeros(3)
        outputs.append(v)
        if owner >= 0 and roots[row]: loss = loss+(v*bar[row]).sum()
    expected = torch.autograd.grad(loss, (xx, ww, bb))
    for actual, ref in zip((y, dx, dw1+dw2, db1+db2), (torch.stack(outputs), *expected)):
        torch.testing.assert_close(actual.double(), ref, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("mode", ["kv", "summary", "both", "none", "zero"])
def test_kv_factored_bank_transpose_independent_roots(library, mode):
    gen = torch.Generator().manual_seed(33)
    x = torch.randn(9, 4, generator=gen)*.2; owners = torch.tensor([0, 1, 0, -1, 1, 0, 1, 0, 1])
    weights = [torch.randn(shape, generator=gen)*.2 for shape in ((3, 8, 4), (3, 8, 3), (3, 3, 4), (3, 2, 4))]
    base, b_t = forward(library, x, owners, weights[0], epsilon=.13)
    low, l_t = forward(library, x, owners, weights[2], epsilon=.13)
    up, u_t = forward(library, low, owners, weights[1])
    summary, s_t = forward(library, x, owners, weights[3], epsilon=.13)
    kv = base+up; bg = torch.randn(kv.shape, generator=gen); sg = torch.randn(summary.shape, generator=gen)
    kr = torch.tensor([True, False, True, True, False, True, True, False, True]) & (mode in ("kv", "both", "zero"))
    sr = torch.tensor([False, True, True, False, True, False, True, True, False]) & (mode in ("summary", "both", "zero"))
    if mode == "zero": bg.zero_(); sg.zero_()
    dbx, db, _, bp = backward(library, b_t, bg, kr)
    dulow, du, _, up_plan = backward(library, u_t, bg, kr)
    dvx, dv, _, _ = backward(library, l_t, dulow, up_plan["connected"])
    dsx, ds, _, sp = backward(library, s_t, sg, sr)
    leaves = [v.double().requires_grad_() for v in (x, *weights)]; xx, b, u, v, s = leaves
    loss = xx.new_zeros(()); kref, sref = [], []
    for row, owner in enumerate(owners.tolist()):
        if owner < 0: kref.append(xx.new_zeros(8)); sref.append(xx.new_zeros(2)); continue
        norm = xx[row]/(.13+xx[row].square().mean()).sqrt()
        ky = b[owner]@norm+u[owner]@(v[owner]@norm); sy = s[owner]@norm
        kref.append(ky); sref.append(sy)
        if kr[row]: loss = loss+(ky*bg[row]).sum()
        if sr[row]: loss = loss+(sy*sg[row]).sum()
    raw = torch.autograd.grad(loss, leaves, allow_unused=True) if loss.requires_grad else [None]*len(leaves)
    expected = [torch.zeros_like(leaf) if grad is None else grad for leaf, grad in zip(leaves, raw)]
    for actual, ref in zip((kv, summary, dbx+dvx+dsx, db, du, dv, ds), (torch.stack(kref), torch.stack(sref), *expected)):
        torch.testing.assert_close(actual.double(), ref, atol=1e-6, rtol=1e-5)
    assert bp["used"].any().item() == (mode in ("kv", "both", "zero"))
    assert sp["used"].any().item() == (mode in ("summary", "both", "zero"))
