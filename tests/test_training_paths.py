"""Frozen optimizer ownership, bounded difference witnesses, actual continuation."""
import copy

import pytest
import torch

from tide import ModelConfig, TopologyConfig, TideModel, place_model, placement_adamw
from scripts.check_training_paths import (THRESHOLDS, VARIANTS, canonical_optimizer, compare,
    checked_optimizer_step, first_differences, first_path_witness, load_canonical_optimizer,
    resolve_variants, run, screen, training_data, validate_restore_record)


def tiny_config(*, hb=False, timing="settle"):
    topology = (TopologyConfig(kind="multiscale", profile="compact-5-dd4", n=2, hubs=1, timing=timing)
                if hb else TopologyConfig(n=2, layers=2))
    return ModelConfig(topology=topology, d=64, memory="K", sharing="region_shared", attention_backend="math")


def test_optimizer_row_mapping_keeps_distinct_slots_and_steps():
    model = TideModel(tiny_config(hb=True))
    optimizer = placement_adamw(model, lr=1e-4, foreach=False)
    # Independent row-dependent contents reveal row swaps and dropped owners.
    for index, parameter in enumerate(model.parameters()):
        position = torch.arange(parameter.numel(), dtype=parameter.dtype).reshape_as(parameter)
        optimizer.state[parameter] = dict(step=torch.tensor(3.), exp_avg=position+index/8,
                                          exp_avg_sq=position.square()+index+1)
    frozen = canonical_optimizer(model, optimizer)
    layout = [(model.graph.nodes[nodes[0]].position+model.graph.nodes[nodes[0]].layer) % 2
              for nodes in model.graph.regions]
    placed = place_model(copy.deepcopy(model), ["cpu:0", "cpu:1"], region_partitions=layout)
    restored = placement_adamw(placed, lr=9e-4, foreach=True)
    load_canonical_optimizer(placed, restored, frozen)
    actual = canonical_optimizer(placed, restored)
    torch.testing.assert_close(actual["state"], frozen["state"], atol=0, rtol=0)
    assert actual["group"] == frozen["group"]
    assert actual["parameter_names"] == frozen["parameter_names"]
    assert actual["group"]["lr"] == 1e-4 and not actual["group"]["foreach"]
    # Corrupt one shard counter: canonicalization must not silently pick one.
    wrapper = next(wrapper for wrapper in placed._shard_registry.values() if len(wrapper.shards) > 1)
    parameter = getattr(wrapper.shards[1], next(iter(wrapper._shapes)))
    restored.state[parameter]["step"] += 1
    with pytest.raises(ValueError, match="inconsistent optimizer scalar"):
        canonical_optimizer(placed, restored)
    restored.state[parameter]["step"] -= 1
    valid = restored.state[parameter]["step"].clone()
    for changed in (valid.double(), valid.reshape(1), int(valid)):
        restored.state[parameter]["step"] = changed
        with pytest.raises(ValueError, match="inconsistent optimizer slot (type|dtype/shape)"):
            canonical_optimizer(placed, restored)
    restored.state[parameter]["step"] = valid


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "foreign"])
def test_optimizer_group_membership_cannot_be_hidden_by_stale_slots(corruption):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), foreach=False)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    frozen = canonical_optimizer(model, optimizer)
    parameters = optimizer.param_groups[0]["params"]
    if corruption == "missing":
        removed = parameters.pop()
        assert removed in optimizer.state
    elif corruption == "duplicate":
        parameters.append(parameters[0])
    else:
        parameters[-1] = torch.nn.Parameter(torch.zeros_like(parameters[-1]))
    for action in (lambda: canonical_optimizer(model, optimizer),
                   lambda: load_canonical_optimizer(model, optimizer, frozen)):
        with pytest.raises(ValueError, match="optimizer group"):
            action()


def test_checked_optimizer_step_counts_only_participating_parameters():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), foreach=False)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    first = checked_optimizer_step(model, optimizer, ["cpu"])
    assert first["checked"] and first["before"] == {"weight": 0, "bias": 0}
    assert first["after"] == {"weight": 1, "bias": 1}
    model.bias.grad = None
    second = checked_optimizer_step(model, optimizer, ["cpu"])
    assert second["active_original_parameters"] == ["weight"]
    assert second["before"] == first["after"]
    assert second["after"] == {"weight": 2, "bias": 1}


@pytest.mark.parametrize("fault", ["no_step", "double_step", "clear_gradients", "no_parameter_gradient"])
def test_claimed_updates_require_gradient_participation_and_one_real_step(monkeypatch, fault):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), foreach=False)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    if fault == "no_step":
        monkeypatch.setattr(optimizer, "step", lambda: None)
    elif fault == "double_step":
        original = optimizer.step
        monkeypatch.setattr(optimizer, "step", lambda: (original(), original()))
    elif fault == "clear_gradients":
        monkeypatch.setattr(optimizer, "step", lambda: optimizer.zero_grad(set_to_none=True))
    else:
        optimizer.zero_grad(set_to_none=True)
    with pytest.raises(RuntimeError, match="(did not advance exactly once|connected model Parameter gradient)"):
        checked_optimizer_step(model, optimizer, ["cpu"])


def test_restore_record_requires_the_current_bundle_and_consistent_process_exit():
    record = dict(state="within_threshold", checks={}, bundle_sha256="current", pid=200)
    validate_restore_record(record, "current", exit_code=0, original_pid=100)
    # Numerical failure is retained as a failure, with the worker's defined exit 2.
    failed = dict(record, state="exact_contract_failed")
    validate_restore_record(failed, "current", exit_code=2, original_pid=100)
    for current, code in ((record, 17), (failed, 0)):
        with pytest.raises(RuntimeError, match="exit code"):
            validate_restore_record(current, "current", exit_code=code, original_pid=100)
    with pytest.raises(RuntimeError, match="current input bundle"):
        validate_restore_record(record, "different", exit_code=0, original_pid=100)
    with pytest.raises(RuntimeError, match="distinct child process"):
        validate_restore_record(record, "current", exit_code=0, original_pid=200)
    with pytest.raises(RuntimeError, match="unknown completion state"):
        validate_restore_record(dict(record, state="running"), "current", exit_code=2, original_pid=100)


def test_support_time_none_and_finite_rounding_have_distinct_witnesses():
    expected = dict(state=dict(weight=torch.tensor([0., 1.]), times=torch.tensor([2, 3])),
                    gradients={"unused":None}, values=torch.tensor([1.]))
    actual = copy.deepcopy(expected)
    actual["state"]["weight"][0] = 1e-8
    actual["state"]["times"][1] = 4
    actual["gradients"]["unused"] = torch.zeros(1)
    actual["values"] += 1e-7
    evidence = compare(actual, expected, THRESHOLDS["fp32"])
    first = evidence["first_differences"]
    assert set(first) == {"support", "time_or_index", "none_gradient", "numeric"}
    assert first["support"]["coordinates"] == [dict(flat_index=0, actual=float(torch.tensor(1e-8)), expected=0.)]
    assert first["time_or_index"]["field"] == "root.state.times"
    assert first["none_gradient"]["field"] == "root.gradients.unused"
    assert evidence["comparison"]["metadata_failures"]
    assert not evidence["comparison"]["threshold_failures"]
    assert screen(actual)["support_sha256"] != screen(expected)["support_sha256"]


def test_witness_coordinate_bound_and_parameter_weight_not_support():
    expected = {"parameters":{"weight":torch.zeros(32)},"values":torch.zeros(64)}
    actual = {"parameters":{"weight":torch.ones(32)},"values":torch.ones(64)}
    first = first_differences(actual, expected)
    assert set(first) == {"numeric"}
    assert len(first["numeric"]["coordinates"]) == 8
    assert VARIANTS["combined"] == dict(region_batch_size=4, checkpoint_regions=True)


def test_internal_witness_stops_at_the_requested_capture_bound(tmp_path):
    config = tiny_config()
    model = TideModel(config)
    optimizer = placement_adamw(model, lr=1e-4, foreach=False)
    anchor = training_data(model, optimizer, None)
    args = (config, "batch4", anchor, torch.zeros(1, 2, 64), "cpu", "fp32", 1, tmp_path)
    result = first_path_witness(*args, witness_elements=1)
    assert result["state"] == "capture_limit"
    assert result["reason"] == "1 captured elements per path"
    assert not (tmp_path/"first-witness.pt").exists()
    for invalid in (0, 20_000_001):
        with pytest.raises(ValueError, match="witness_elements"):
            first_path_witness(*args, witness_elements=invalid)


def test_time_witness_keeps_int64_above_float_exact_range():
    expected = {"state":{"times":torch.tensor([2**53+3]), "weight":torch.tensor([1.])}}
    actual = {"state":{"times":torch.tensor([2**53+4]), "weight":torch.tensor([0.])}}
    first = first_differences(actual, expected)["time_or_index"]
    assert first["coordinates"] == [dict(flat_index=0,actual=2**53+4,expected=2**53+3)]
    assert type(first["coordinates"][0]["expected"]) is int
    assert first["paired_key_weights"] == [dict(flat_index=0,actual=0.,expected=1.)]


def test_named_execution_candidates_preserve_builtin_and_semantic_meanings():
    variants = resolve_variants({"reception":{"checkpoint_reception":True},
                                 "prefix":{"ssd_summary_backend":"prefix"}})
    assert variants["baseline"] == VARIANTS["baseline"]
    assert variants["combined"] == VARIANTS["combined"]
    assert variants["reception"] == dict(region_batch_size=1,checkpoint_regions=False,checkpoint_reception=True)
    with pytest.raises(ValueError,match="built-in meanings"):
        resolve_variants({"baseline":{"checkpoint_regions":True}})
    with pytest.raises(ValueError,match="execution-option"):
        resolve_variants({"changed_math":{"budget":2}})


def test_real_anchor_repeat_update_continuation_and_restore(tmp_path):
    result = run(tiny_config(), device="cpu", precision="fp64", candidates=["baseline", "recompute"],
                 batch=1, length=2, steps=2, chunk=1, directory=tmp_path, new_process=False, witness=False)
    assert result["state"] in {"within_threshold", "finite_deviation"}
    assert result["anchor_optimizer"]["parameter_states"] > 0
    assert result["anchor_optimizer"]["steps"] == [1]
    assert len(result["candidates"]) == 2
    for case in result["candidates"]:
        assert case["checks"]["fixed_start"]["comparison"]["exact"]
        assert case["checks"]["start_unchanged_after_vjp"]["comparison"]["exact"]
        assert case["checks"]["same_owner_vjp_repeat"]["comparison"]["exact"]
        assert case["restore"]["checks"]["saved_data"]["comparison"]["exact"]
        assert case["restore"]["checks"]["next_forward"]["comparison"]["exact"]
        assert case["restore"]["actual_update"]
        assert len(case["updates"]) == 2
    anchor = torch.load(tmp_path/"anchor.pt", weights_only=False)["anchor"]
    assert anchor["state"].position == 2
    assert anchor["optimizer"]["state"]
    assert all(slot["step"] == 1 for slot in anchor["optimizer"]["state"].values())
    restored = torch.load(tmp_path/"baseline/restore-input.pt", weights_only=False)
    assert restored["saved"]["state"].position == 6
    assert restored["updated"]["state"].position == 8
    assert all(slot["step"] == 4 for slot in restored["updated"]["optimizer"]["state"].values())


def test_fresh_process_restores_the_same_frozen_model_and_really_updates(tmp_path):
    result = run(tiny_config(), device="cpu", precision="fp32", candidates=["baseline"],
                 batch=1, length=2, steps=1, chunk=1, directory=tmp_path, new_process=True, witness=False)
    restore = result["candidates"][0]["restore"]
    assert restore["new_process"] and restore["pid"] != restore["original_process_pid"]
    assert restore["checks"]["saved_data"]["comparison"]["exact"]
    assert restore["checks"]["next_forward"]["comparison"]["exact"]
    assert restore["actual_update"] and restore["config_exact"]
    assert restore["source_exact"]
    assert "auxiliary harness modules" in restore["source_identity_scope"]
    assert restore["optimizer_update"]["active_original_parameters"]
    assert restore["optimizer_update"]["checked"]


def test_logical_cpu_placement_uses_the_common_nonempty_optimizer(tmp_path):
    result = run(tiny_config(hb=True), device="cpu:0", precision="fp32",
        candidates=["baseline", "placement_serial", "placement_per_device"],
        placement_devices=["cpu:0", "cpu:1"], batch=1, length=2, steps=1, chunk=1,
        directory=tmp_path, new_process=False, witness=False)
    assert result["state"] in {"within_threshold", "finite_deviation"}
    for case in result["candidates"][1:]:
        assert case["checks"]["fixed_start"]["comparison"]["exact"]
        assert case["checks"]["vjp1_vs_baseline"]["comparison"]["metadata_failures"] == []
        assert case["restore"]["checks"]["saved_data"]["comparison"]["exact"]
        assert case["placement"]["execution"]["policy"] == case["name"].removeprefix("placement_")


def test_bf16_pure_batch_and_recompute_combination_continues_and_restores(tmp_path):
    config = tiny_config(hb=True, timing="timed")
    assert config.local_attention_backend is None
    result = run(config, device="cpu", precision="bf16",
        candidates=["baseline", "batch4", "recompute", "combined"],
        batch=1, length=2, steps=1, chunk=1, directory=tmp_path, new_process=False, witness=True)
    # Mechanism completion does not erase finite/support/time comparisons.
    for case in result["candidates"]:
        assert case["config"]["local_attention_backend"] is None
        assert case["checks"]["fixed_start"]["comparison"]["exact"]
        assert case["restore"]["checks"]["saved_data"]["comparison"]["exact"]
        assert case["restore"]["checks"]["next_forward"]["comparison"]["exact"]
        assert not case["restore"]["checks"]["actual_update"]["comparison"]["nonfinite"]
        assert case["restore"]["actual_update"]
    assert result["candidates"][-1]["config"]["region_batch_size"] == 4
    assert result["candidates"][-1]["config"]["checkpoint_regions"]
