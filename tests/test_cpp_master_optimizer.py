"""Independent Torch master updates, exact native decode, and format truth anchors."""
import struct
import pytest
import torch
import _tide_native as core
from tidegraph.precision import FP32MasterOptimizer
from test_cpp_checkpoint import _state_snapshot, _assert_state


def fixture(kind, scale=16.):
    values = [torch.tensor([1., -2.], dtype=torch.float16, requires_grad=True),
              torch.tensor([.5], dtype=torch.float16, requires_grad=True)]
    registry = core.ParameterRegistry()
    registry.add("a", values[0]); registry.add("alias", values[0]); registry.add("z", values[1])
    group = core.OptimizerGroup(); group.parameters = ["alias", "z"]
    group.lr = .01; group.weight_decay = .1
    policy = core.OptimizerPrecision(); policy.fp32_masters = True; policy.loss_scale = scale
    optimizer = (core.SGD if kind == "sgd" else core.AdamW)(registry, [group], policy)
    return registry, values, optimizer


def loss(values, step):
    # First two objectives omit z, then connect z with a numerical zero.
    result = (values[0].float()*values[0].new_tensor([.5, -.25]).float()).sum()
    return result if step < 2 else result + values[1].float().sum()*0


@pytest.mark.parametrize("kind", ["sgd", "adamw"])
def test_native_half_master_updates_exact_checkpoint_and_absent_zero(kind, tmp_path):
    registry, values, optimizer = fixture(kind)
    reference = [p.detach().clone().requires_grad_() for p in values]
    expected = FP32MasterOptimizer(reference, optimizer=kind, lr=.01, weight_decay=.1, loss_scale=16.)
    for step in range(3):
        optimizer.zero_grad(); expected.zero_grad()
        optimizer.backward(loss(values, step)); expected.backward(loss(reference, step))
        optimizer.step(); expected.step()
        for actual, wanted in zip(values, reference):
            torch.testing.assert_close(actual, wanted, atol=0, rtol=0)
        for name, master in zip(("a", "z"), expected.masters):
            torch.testing.assert_close(optimizer.masters()[name], master, atol=1e-6, rtol=1e-5)
        if kind == "adamw":
            assert ("z" in optimizer.state()) == (step >= 2)
            for s in optimizer.state().values():
                assert s.exp_avg.dtype == s.exp_avg_sq.dtype == torch.float32
    path = tmp_path/"half.tidenck"
    core.save_checkpoint(str(path), registry, optimizer, "half-v1")
    assert struct.unpack_from("<I", path.read_bytes(), 8)[0] == 2
    rr, rv, ro = fixture(kind, scale=1.)
    core.load_checkpoint(str(path), rr, ro, "half-v1")
    assert ro.precision().loss_scale == 16.
    _assert_state(_state_snapshot(ro), _state_snapshot(optimizer))
    for name, value in optimizer.masters().items():
        torch.testing.assert_close(ro.masters()[name], value, atol=0, rtol=0)
    for opt, params in ((optimizer, values), (ro, rv)):
        opt.zero_grad(); opt.backward(loss(params, 3)); opt.step()
    for a, b in zip(values, rv):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    _assert_state(_state_snapshot(ro), _state_snapshot(optimizer))


def test_native_master_refuses_undeclared_scaling_nonfinite_and_incoherent_save(tmp_path):
    registry, values, opt = fixture("adamw")
    with pytest.raises((ValueError, RuntimeError), match="explicit FP32 masters"):
        core.AdamW(registry)
    values[0].grad = torch.ones_like(values[0])
    with pytest.raises((ValueError, RuntimeError), match="declare gradient scaling"):
        opt.step()
    opt.zero_grad(); opt.backward(loss(values, 0))
    values[1].grad = torch.full_like(values[1], float("inf"))
    before = {n: t.clone() for n, t in opt.masters().items()}
    with pytest.raises((ValueError, RuntimeError), match="nonfinite"):
        opt.step()
    assert not opt.state()
    for n, t in opt.masters().items():
        torch.testing.assert_close(t, before[n], atol=0, rtol=0)
    opt.masters()["a"].add_(1)
    path = tmp_path/"must-not-exist"
    with pytest.raises((ValueError, RuntimeError), match="master and payload"):
        core.save_checkpoint(str(path), registry, opt)
    assert not path.exists()


def fnv(data):
    value = 14695981039346656037
    for byte in data:
        value = ((value ^ byte)*1099511628211) % 2**64
    return struct.pack("<Q", value)


def test_valid_checksum_master_payload_mismatch_refuses_before_mutation(tmp_path):
    registry, values, opt = fixture("adamw")
    opt.backward(loss(values, 0)); opt.step()
    path = tmp_path/"source"; core.save_checkpoint(str(path), registry, opt)
    body = path.read_bytes()[:-8]
    master = opt.masters()["a"].tolist()
    # The documented v2 FP32 1D tensor layout, independently constructed.
    pattern = bytes([0, 1, 1])+struct.pack("<qQ2f", 2, 2, *master)
    assert body.count(pattern) == 1
    body = body.replace(pattern, pattern[:-8]+struct.pack("<2f", master[0]+1., master[1]))
    bad = tmp_path/"incoherent"; bad.write_bytes(body+fnv(body))
    rr, rv, ro = fixture("adamw", scale=1.)
    before = [p.detach().clone() for p in rv]
    before_masters = {n: t.clone() for n, t in ro.masters().items()}
    with pytest.raises((RuntimeError, ValueError), match="master and payload"):
        core.load_checkpoint(str(bad), rr, ro)
    assert not ro.state() and ro.precision().loss_scale == 1.
    for actual, saved in zip(rv, before):
        torch.testing.assert_close(actual, saved, atol=0, rtol=0)
    for name, saved in before_masters.items():
        torch.testing.assert_close(ro.masters()[name], saved, atol=0, rtol=0)


def test_independent_legacy_v1_weights_decode_and_v2_half_tag_gate(tmp_path):
    # Construct the previous wire format independently of the production codec.
    string = lambda x: struct.pack("<Q", len(x))+x
    tensor = bytes([0, 1, 1])+struct.pack("<qQ2f", 2, 2, .25, -.5)
    body = b"TIDENCK1"+struct.pack("<IB", 1, 1)+string(b"legacy")
    body += struct.pack("<QQ", 1, 1)+string(b"weight")+tensor+bytes([0])
    path = tmp_path/"legacy"; path.write_bytes(body+fnv(body))
    r = core.ParameterRegistry(); p = torch.zeros(2); r.add("weight", p)
    core.load_checkpoint(str(path), r, None, "legacy")
    torch.testing.assert_close(p, torch.tensor([.25, -.5]), atol=0, rtol=0)
    # A valid checksum cannot authorize a half tag in the old schema.
    raw = body.replace(tensor, bytes([0, 4, 1])+struct.pack("<qQ2e", 2, 2, .25, -.5))
    path = tmp_path/"bad-half-v1"; path.write_bytes(raw+fnv(raw))
    with pytest.raises((ValueError, RuntimeError), match="requires native schema v2"):
        core.load_checkpoint(str(path), r)
