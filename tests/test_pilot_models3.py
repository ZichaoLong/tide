"""Independent dispatch/gradient, ownership, continuation and init witnesses."""
import copy
import io

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from pilotlab.models import build_model as build_previous
from pilotlab.models3 import (INITIALIZATION_VERSIONS, TIMED_LAST_NONEMPTY, SparseMoE, build_model,
                              count_owner_activity, detach_state, health_snapshot,
                              last_nonempty_readout, parameter_counts, parameter_snapshot, tensor_health)


@pytest.fixture(autouse=True, scope="module")
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


MOE = dict(family="moe", d=32, heads=2, layers=2, ffn_width=48, window=4, experts=4, top_k=2)
TIDE = dict(family="tide", config=dict(topology=dict(kind="fc", n=2, layers=2),
            d=64, memory="K", sharing="region_shared", budget=1, attention_backend="math"))


def test_sparse_dispatch_never_evaluates_unselected_rows_or_experts():
    module = SparseMoE(4, 8, 3, 1)
    with torch.no_grad():
        module.router.weight.copy_(torch.tensor([[1., 0, 0, 0], [0, 1., 0, 0], [-1., -1, 0, 0]]))
    calls = [[], [], []]
    hooks = [expert.register_forward_pre_hook(lambda m, args, i=i: calls[i].append(args[0].shape[0]))
             for i, expert in enumerate(module.experts)]
    x = torch.tensor([[[2., 0, 0, 0], [0, 2., 0, 0], [3., 0, 0, 0], [0, 3., 0, 0]]])
    _, auxiliary, work = module(x, record_work=True)
    for hook in hooks:
        hook.remove()
    assert calls == [[2], [2], []]
    assert work["dispatch_rows_by_expert"] == [2, 2, 0]
    assert work["selected_expert_parameters_per_token"].tolist() == [96]*4
    assert auxiliary.isfinite()


def test_sparse_dispatch_matches_independent_dense_reference_and_gradients():
    torch.manual_seed(46)
    actual = SparseMoE(8, 12, 4, 2).double()
    reference = copy.deepcopy(actual)
    x = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    output, auxiliary, _ = actual(x)
    flat = xr.reshape(-1, 8)
    scores = flat @ reference.router.weight.T
    selected_scores, indices = scores.topk(2, -1)
    gates = selected_scores.softmax(-1)
    all_outputs = torch.stack([expert(flat) for expert in reference.experts], 1)
    selected = all_outputs.gather(1, indices[..., None].expand(-1, -1, 8))
    expected = (selected * gates[..., None]).sum(1).reshape_as(xr)
    density = F.one_hot(indices, 4).float().mean((0, 1))
    expected_aux = 4 * (density * scores.softmax(-1).mean(0)).sum()
    weights = torch.randn_like(output)
    (output.mul(weights).sum() + .03*auxiliary).backward()
    (expected.mul(weights).sum() + .03*expected_aux).backward()
    torch.testing.assert_close(output, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(x.grad, xr.grad, rtol=1e-11, atol=1e-12)
    for (name, parameter), (other_name, other) in zip(actual.named_parameters(), reference.named_parameters()):
        assert name == other_name
        torch.testing.assert_close(parameter.grad, other.grad, rtol=1e-10, atol=1e-12)


def test_owner_activity_uses_per_token_union_and_private_adapters():
    from tide.parameters import NodeLinear, NodeVector
    module = NodeLinear(3, 4, 6, groups=[0, 0, 1], rank=2, bias=True)
    selected = torch.tensor([[True, False, True], [True, True, False], [False, True, False]])
    # One base+bias is 30; each owner's two private adapters are 20.
    assert count_owner_activity(module, selected).tolist() == [70, 100, 50]
    vector = NodeVector(3, 5, groups=[0, 0, 1])
    assert count_owner_activity(vector, selected).tolist() == [5, 10, 5]


@pytest.mark.parametrize("family", ["transformer", "moe"])
def test_actual_activity_includes_shared_norm_attention_router_and_tied_head(family):
    model = build_model(dict(MOE, family=family), 37)
    output = model(torch.tensor([[1, 2], [3, 4]]), record_work=True)
    measured = output.work["activity"]
    counts = parameter_counts(model)
    assert measured["qualified"]
    assert counts["vocabulary_parameters"] == 37*32
    assert counts["whole_model_parameters"] == sum(p.numel() for p in model.parameters())
    expected = counts["backbone_total_parameters"]
    if family == "moe":
        # Half the independent expert parameters participate per token, while
        # every attention, router, norm parameter participates for every token.
        expert_count = sum(p.numel() for block in model.blocks for p in block.moe.experts.parameters())
        expected -= expert_count//2
    assert measured["mean_active_backbone_parameters"] == expected
    assert measured["mean_active_whole_model_parameters"] == expected + 37*32


@pytest.mark.parametrize("spec", [TIDE, dict(TIDE, config=dict(TIDE["config"], memory="S")),
                                dict(family="transformer", d=32, heads=2, layers=2, ffn_width=48, window=4)])
def test_versioned_default_preserves_previous_initialization_exactly(spec):
    torch.manual_seed(778)
    previous = build_previous(spec, 31)
    torch.manual_seed(778)
    current = build_model(spec, 31)
    assert current.initialization_recipe == INITIALIZATION_VERSIONS[spec["family"]]
    assert current.state_dict().keys() == previous.state_dict().keys()
    for name, value in previous.state_dict().items():
        torch.testing.assert_close(value, current.state_dict()[name], rtol=0, atol=0)
    tokens = torch.tensor([[1, 2, 3]])
    torch.testing.assert_close(previous(tokens).logits, current(tokens, record_work=True).logits, rtol=0, atol=0)
    with pytest.raises(ValueError, match="unsupported initialization"):
        build_model(spec, 31, "unregistered-rescale")


def test_tide_capture_has_no_parameter_or_backward_recomputation_effect():
    spec = copy.deepcopy(TIDE)
    spec["config"].update(checkpoint_regions=True, ffn_packing="power2", region_batch_size=4)
    plain = build_model(spec, 31)
    measured = copy.deepcopy(plain)
    assert measured.core.memories["K"]._round3_activity_meter.model is measured
    tokens = torch.tensor([[1, 2, 3], [3, 4, 5]])
    first, second = plain(tokens), measured(tokens, record_work=True)
    torch.testing.assert_close(first.logits, second.logits, rtol=0, atol=0)
    assert second.work["activity"]["qualified"]
    first.logits.square().mean().backward()
    second.logits.square().mean().backward()
    for p, q in zip(plain.parameters(), measured.parameters()):
        if p.grad is None:
            assert q.grad is None
        else:
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
    assert measured._activity_meter.capture is None


@pytest.mark.parametrize("memory", ["K", "S", "X", "X_reverse"])
def test_tide_full_activity_covers_memory_prepare_and_all_optional_paths(memory):
    spec = dict(family="tide", config=dict(
        topology=dict(kind="exchange", n=2, hubs=1, depth=1, rounds=1),
        d=64, memory=memory, aggregation="incoming_light_64", sharing="layer_shared",
        memory_sharing="region_shared", budget=1, attention_backend="math",
        region_batch_size=4, ffn_packing="power2", reception_qkv_backend="fused"))
    model = build_model(spec, 31)
    output = model(torch.tensor([[1, 2], [3, 4]]), record_work=True)
    activity = output.work["activity"]
    assert activity["qualified"], activity
    assert 0 < activity["always_active_backbone_parameters"] <= activity["min_active_backbone_parameters"]
    assert activity["max_active_backbone_parameters"] <= activity["backbone_total_parameters"]
    assert activity["measured_external_tokens"] == 4
    output.logits.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_timed_activity_unions_ticks_within_each_external_token():
    from tide.parameters import NodeOwners
    spec = dict(family="tide", config=dict(topology=dict(kind="multiscale", profile="compact-5-dd4",
                n=2, timing="timed", period=4), d=64, memory="K", sharing="independent"))
    model = build_model(spec, 31)
    meter = model._activity_meter
    n = len(model.core.memory_ids["K"])
    meter.begin(torch.ones(2, 2, dtype=torch.int64))
    control = torch.zeros(n, 2, 8)
    control[0, 0, 0] = control[0, 0, 3] = 1  # same owner twice, same token
    control[1, 1, 4] = 1                    # other sample, other token
    meter._record("K", NodeOwners.from_static(tuple(range(n)), device="cpu"), control)
    measured = meter.end()
    dynamic_one = sum(sum(p[0].numel() for p in module.parameters(recurse=False))
                      for module in meter.dynamic["K"])
    assert measured["qualified"]
    assert measured["semantic_full_owner_tick_rows"] == 3
    assert measured["min_active_backbone_parameters"] == meter.always_count
    assert measured["max_active_backbone_parameters"] == meter.always_count + dynamic_one
    assert measured["mean_active_backbone_parameters"] == meter.always_count + dynamic_one/2


def test_last_nonempty_policy_against_independent_slot_oracle_and_empty_gradients():
    values = torch.arange(32, dtype=torch.float64).reshape(2, 8, 2).requires_grad_()
    strengths = torch.tensor([[1., 0, 2, 0, 0, 0, 0, 0], [0, 3, 0, 4, 5, 0, 0, 0]])
    actual, positions = last_nonempty_readout(values, strengths, 4)
    expected = torch.zeros(2, 2, 2, dtype=torch.float64)
    expected_positions = []
    for batch in range(2):
        observed = []
        for token in range(2):
            selected = -1
            for tick in range(4):
                if strengths[batch, token*4+tick] > 0:
                    selected = tick
            observed.append(selected)
            if selected >= 0:
                expected[batch, token] = values.detach()[batch, token*4+selected]
        expected_positions.append(observed)
    assert positions.tolist() == expected_positions == [[2, -1], [3, 0]]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    expected_gradient = torch.zeros_like(values)
    expected_gradient[0, 2] = expected_gradient[1, 3] = expected_gradient[1, 4] = 1
    torch.testing.assert_close(values.grad, expected_gradient, rtol=0, atol=0)


def test_versioned_timed_readout_is_causal_nonempty_and_state_partition_invariant():
    from pilotlab.catalog3 import generate_subjects
    row = next(row for row in generate_subjects() if row["group"] == "multiscale_timed"
               and row["model"]["config"]["d"] == 64 and row["model"]["config"]["memory"] == "K"
               and row["coverage_variant"] == "group5")
    torch.manual_seed(208)
    model = build_model(row["model"], 31)
    assert model.readout_policy == TIMED_LAST_NONEMPTY
    tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    changed = tokens.clone()
    changed[:, 2:] = 9
    with torch.no_grad():
        full = model(tokens, record_work=True)
        altered = model(changed, record_work=True)
        prefix = model(tokens[:, :2], record_work=True)
        suffix = model(tokens[:, 2:], prefix.state, record_work=True)
    torch.testing.assert_close(full.logits[:, :2], altered.logits[:, :2], rtol=3e-4, atol=3e-5)
    torch.testing.assert_close(full.logits, torch.cat((prefix.logits, suffix.logits), 1), rtol=3e-4, atol=3e-5)
    assert full.work["empty_output_positions"] == 0
    assert full.logits.abs().max() > 0
    assert suffix.state.position == tokens.shape[1]
    assert [row[:2] for row in full.work["selected_tick_in_period"]] == prefix.work["selected_tick_in_period"]
    assert [row[2:] for row in full.work["selected_logical_time"]] == suffix.work["selected_logical_time"]
    for batch in range(2):
        for token, observed in enumerate(full.work["selected_logical_time"][batch]):
            assert token*6+1 <= observed < (token+1)*6+1
    assert full.work["readout_inserts_flush"] is False
    # Keep the former policy as a distinct, inspectable identity and retain its
    # failure witness: a period longer than the longest path has an empty last tick.
    previous_spec = copy.deepcopy(row["model"])
    previous_spec.pop("readout_policy")
    previous = build_model(previous_spec, 31)
    previous.load_state_dict(model.state_dict())
    old = previous(tokens, record_work=True)
    assert old.work["empty_output_positions"] == tokens.numel()
    assert torch.count_nonzero(old.logits) == 0
    learned = model(tokens)
    F.cross_entropy(learned.logits.flatten(0, 1), (tokens+1).flatten()).backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad) > 0
               for name, p in model.named_parameters() if name.startswith("core."))


def test_activity_fails_closed_for_uncovered_parameter():
    from pilotlab.models3 import _TideActivity
    model = build_model(TIDE, 31)
    model.core.unexpected = nn.Parameter(torch.ones(3))
    model._activity_meter = _TideActivity(model)
    result = model(torch.tensor([[1, 2]]), record_work=True)
    assert not result.work["activity"]["qualified"]
    assert any("uncovered" in reason for reason in result.work["activity"]["reasons"])


@pytest.mark.parametrize("build", [build_previous, build_model])
def test_timed_head_counts_a_weak_sender_whose_physical_strength_rounds_to_zero(build, monkeypatch):
    from tide.model import WeakLogResult

    spec = dict(family="tide", readout_policy=TIMED_LAST_NONEMPTY, config=dict(
        topology=dict(kind="multiscale", profile="compact-7", n=2, timing="timed", period=8),
        d=64, memory="K", allocation="shannon", shannon_support_policy="weak-log-v1"))
    model = build(spec, 31)
    period = model.core.graph.period
    codes = torch.zeros(1, 1, period)
    codes[:, :, -1] = -120.
    physical = torch.zeros_like(codes)
    result = WeakLogResult(torch.ones(1, 1, period, 64), physical, torch.arange(period),
                           None, strength_coordinates=codes)
    monkeypatch.setattr(model.core, "forward", lambda *args, **kw: result)
    work = model(torch.tensor([[7]]), record_work=True).work
    assert work["empty_output_positions"] == 0
    assert work["empty_logical_output_slots"] == period - 1
    if build is build_model:
        assert work["selected_tick_in_period"] == [[period - 1]]


def test_moe_causal_continuation_and_fresh_owner_optimizer_resume():
    torch.manual_seed(941)
    model = build_model(MOE, 31)
    tokens = torch.tensor([[1, 3, 4, 2, 6, 5]])
    changed = tokens.clone()
    changed[:, 3:] = 9
    with torch.no_grad():
        full = model(tokens)
        altered = model(changed)
        first = model(tokens[:, :3])
        second = model(tokens[:, 3:], first.state)
    torch.testing.assert_close(full.logits[:, :3], altered.logits[:, :3], rtol=0, atol=0)
    torch.testing.assert_close(full.logits, torch.cat((first.logits, second.logits), 1), rtol=1e-5, atol=1e-6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)

    def step(owner, opt, tok, state):
        opt.zero_grad(set_to_none=True)
        result = owner(tok, state)
        loss = result.logits.square().mean() + .01*result.auxiliary_loss
        loss.backward()
        opt.step()
        return detach_state(result.state)

    state = step(model, optimizer, tokens[:, :3], None)
    checkpoint = io.BytesIO()
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), state=state), checkpoint)
    checkpoint.seek(0)
    saved = torch.load(checkpoint, weights_only=False)
    torch.manual_seed(123)
    restored = build_model(MOE, 31)
    restored.load_state_dict(saved["model"])
    recovered_optimizer = torch.optim.AdamW(restored.parameters(), lr=.001)
    recovered_optimizer.load_state_dict(saved["optimizer"])
    step(model, optimizer, tokens[:, 3:], state)
    step(restored, recovered_optimizer, tokens[:, 3:], saved["state"])
    for first, second in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_health_reports_real_update_and_does_not_fail_zero_gradients():
    model = build_model(MOE, 31)
    before = parameter_snapshot(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    output = model(torch.tensor([[1, 2]]))
    output.logits.square().mean().backward()
    optimizer.step()
    report = health_snapshot(model, before)
    assert report["parameters"]["embedding.weight"]["update"]["rms"] > 0
    assert "qualification" not in report


def test_health_norm_does_not_create_its_own_overflow_or_underflow():
    for value in (1e30, 1e-30):
        stats = tensor_health(torch.tensor([value, -value]))
        assert stats["nonfinite"] == 0
        assert stats["rms"] == pytest.approx(value, rel=1e-6)
