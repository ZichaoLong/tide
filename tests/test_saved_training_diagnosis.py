"""Small owned fixtures verify streaming comparisons and independent AdamW replay."""
import copy
from decimal import Decimal
import hashlib
import json

import pytest
import torch

from scripts.diagnose_saved_training import (
    Inputs, TOLERANCE, adamw_decimal, adamw_fp64, anchor_bias_coordinates,
    compare_saved, flat_chunk, replay_coordinate, run, write_outputs,
)
from scripts.qualify_precision import comparison


def test_streaming_matches_original_threshold_direction_and_keeps_int64_support():
    expected = dict(
        gradients={"weight": torch.arange(12, dtype=torch.float32).reshape(3, 4).T},
        state=dict(times=torch.tensor([2**53 + 3, 2**53 + 5]), weight=torch.tensor([0., 1.])),
        input_gradient=None,
    )
    actual = copy.deepcopy(expected)
    actual["gradients"]["weight"][0, 0] = 2e-6
    actual["gradients"]["weight"][1, 1] += .04
    actual["state"]["times"][1] += 1
    actual["state"]["weight"][0] = 1e-12
    # A parameter called weight must not be mistaken for message support.
    actual["gradients"]["weight"][0, 0] = -2e-6
    original = comparison(actual, expected, TOLERANCE)
    for chunk in [1, 5, 100]:
        result = compare_saved(actual, expected, chunk_elements=chunk)
        assert result["threshold_failures"] == original["threshold_failures"]
        assert result["metadata_failures"] == original["metadata_failures"]
        assert result["support_differences"] == ["root.state.weight"]
        clock = next(x for x in result["changed_fields"] if x["field"] == "root.state.times")
        assert clock["top_coordinates"][0]["actual"] == 2**53 + 6
        assert clock["top_coordinates"][0]["reference"] == 2**53 + 5
    value = expected["gradients"]["weight"]
    assert not value.is_contiguous()
    assert torch.equal(flat_chunk(value, 3, 5), value.reshape(-1)[3:8])


def test_nonfinite_metadata_and_top_ties_are_explicit_and_json_finite():
    a = {"x": torch.tensor([2e-6, 2e-6, float("nan")]), "input_gradient": None}
    b = {"x": torch.zeros(3), "input_gradient": torch.zeros(1)}
    result = compare_saved(a, b, chunk_elements=1, top_k=2)
    assert result["state"] == "nonfinite"
    assert "root.input_gradient.type" in result["metadata_failures"]
    row = result["changed_fields"][-1]
    assert row["nonfinite_elements"] == 1
    assert [point["flat_index"] for point in row["top_coordinates"]] == [0, 1]
    assert row["nonfinite_coordinates"][0]["actual"] == "nan"
    json.dumps(result, allow_nan=False)


GROUP = dict(lr=1e-4, betas=(.9, .999), eps=1e-8, weight_decay=.01, amsgrad=False, maximize=False)


def state(parameter=.2, first=1e-9, second=1e-18, step=3, name="bias"):
    return dict(parameters={name: torch.tensor([parameter], dtype=torch.float64)},
                optimizer=dict(group=dict(GROUP), state={name: dict(
                    exp_avg=torch.tensor([first], dtype=torch.float64),
                    exp_avg_sq=torch.tensor([second], dtype=torch.float64), step=torch.tensor(float(step)))}))


def actual_update(before, gradient, name="bias"):
    """A genuine tiny Torch update supplies an independent test observation."""
    parameter = torch.nn.Parameter(before["parameters"][name].clone())
    optimizer = torch.optim.AdamW([parameter], foreach=False, **before["optimizer"]["group"])
    optimizer.state[parameter] = copy.deepcopy(before["optimizer"]["state"][name])
    parameter.grad = torch.tensor([gradient], dtype=torch.float64)
    optimizer.step()
    result = dict(parameters={name: parameter.detach().clone()},
                  optimizer=dict(group=dict(before["optimizer"]["group"]), state={name: copy.deepcopy(optimizer.state[parameter])}),
                  gradients={name: parameter.grad.clone()})
    return result


def test_decimal_and_fp64_replay_match_real_step_and_distinguish_history_gradient():
    before_b = state()
    before_a = state(parameter=.2000001, first=1.1e-9, second=1.3e-18)
    after_b = actual_update(before_b, .7e-9)
    after_a = actual_update(before_a, -.4e-9)
    result = replay_coordinate(before_b, before_a, after_b, after_a, "bias", 0)
    assert result["state"] == "observed_adjacent_update_replay"
    assert abs(float(result["observed_minus_predicted_difference"])) < 1e-15
    delta = Decimal(result["predicted_parameter_difference"])
    assert abs(sum(map(Decimal, result["ordered_contributions"].values())) - delta) < Decimal("1e-30")
    assert all(Decimal(value) != 0 for value in result["ordered_contributions"].values())
    assert result["gradient_magnitudes_at_most_eps"]
    assert abs(float(result["current_gradient_secant_gain"])) > 1000
    assert result["epsilon_ratios"]["actual"]["abs_gradient_over_eps"] < 1
    for owner, observed in [("reference", after_b), ("actual", after_a)]:
        assert abs(result["predictions"][owner]["fp64"]["parameter"] - observed["parameters"]["bias"].item()) < 1e-15
    with pytest.raises(ValueError, match="adjacent"):
        replay_coordinate(state(step=1), state(step=1), after_b, after_a, "bias", 0)
    invalid = dict(GROUP, amsgrad=True)
    with pytest.raises(ValueError, match="AMSGrad"):
        adamw_decimal(.2, 1e-9, 0., 0., 1, invalid)
    with pytest.raises(ValueError, match="nonfinite"):
        adamw_fp64(.2, float("nan"), 0., 0., 1, GROUP)


def test_common_anchor_prediction_has_no_observed_update_and_targets_bias_fields():
    name = "regional.26.second.bias"
    before = state(first=0., second=0., step=1, name=name)
    b, a = {"gradients": {name: torch.tensor([1e-9])}}, {"gradients": {name: torch.tensor([-1e-9])}}
    chosen, ranks = anchor_bias_coordinates(before, b, a)
    assert chosen == [(name, 0)]
    assert ranks[0]["state"] == "counterfactual_coordinates_ranked"
    assert ranks[1]["state"] == "field_absent"
    result = replay_coordinate(before, before, b, a, name, 0, counterfactual=True)
    assert result["state"] == "anchor_vjp_counterfactual"
    assert result["observed_after"] is None and result["actual_parameter_difference"] is None
    assert abs(float(result["predicted_parameter_difference"])) > 1e-6


def fixture(directory):
    directory.mkdir()
    record = dict(schema="tide-execution-qualification-v1", precision="fp32", tolerance=TOLERANCE,
                  state="finite_deviation", qualified=False, source={"source_commit": "owned-tiny-fixture"},
                  artifacts=[], candidates=[dict(name="baseline"), dict(name="weight_access", checks={}, updates=[])])
    anchor = state(first=0., second=0., step=1)
    pairs = {}
    for owner, gradient in [("baseline", 1e-9), ("weight_access", -1e-9)]:
        (directory / owner).mkdir()
        vjp = {"gradients": {"bias": torch.tensor([gradient], dtype=torch.float64)}}
        # update-0 is deliberately missing. update-1/2 are true adjacent steps.
        update0 = actual_update(anchor, gradient)
        update1 = actual_update(update0, gradient)
        update2 = actual_update(update1, gradient)
        for phase, value in [("vjp1", vjp), ("update-1", update1), ("update-2", update2)]:
            pairs[owner, phase] = value
            torch.save(value, directory / owner / (phase + ".pt"))
    torch.save({"training": anchor, "extra": {}, "inputs": []}, directory / "anchor.pt")
    for path in sorted(directory.rglob("*.pt")):
        record["artifacts"].append(dict(path=str(path.relative_to(directory)), bytes=path.stat().st_size,
                                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(), retained=True))
    for phase in ["vjp1", "update-1", "update-2"]:
        check = {"comparison": comparison(pairs["weight_access", phase], pairs["baseline", phase], TOLERANCE)}
        if phase == "vjp1":
            record["candidates"][1]["checks"]["vjp1_vs_baseline"] = check
        else:
            record["candidates"][1]["updates"].append(dict(index=int(phase[-1]), vs_baseline=check))
    (directory / "result.json").write_text(json.dumps(record))
    return record


def test_small_mmap_end_to_end_keeps_inputs_missing_history_and_original_failures(tmp_path):
    source = tmp_path / "source"
    original = fixture(source)
    before = {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}
    report, witness = run(source, tmp_path / "output", chunk_elements=1, verify_sha=True)
    assert report["state"] == "diagnosis_complete"
    assert report["original_state"] == "finite_deviation" and report["original_qualified"] is False
    assert report["all_original_threshold_field_sets_match"]
    assert report["all_original_failure_field_sets_match"]
    assert report["update0_presence"] == {"baseline": False, "weight_access": False}
    assert all(row["content_sha256_verified"] for row in report["input_files"])
    phases = {row["phase"]: row for row in witness["phases"]}
    assert all(row["state"] == "missing_previous_update" for row in phases["update-1"]["adamw"])
    assert phases["update-2"]["adamw"][0]["state"] == "observed_adjacent_update_replay"
    assert abs(float(phases["update-2"]["adamw"][0]["observed_minus_predicted_difference"])) < 1e-15
    assert before == {path: path.read_bytes() for path in before}
    assert sum(path.stat().st_size for path in (tmp_path / "output").iterdir()) < 100_000
    with pytest.raises(MemoryError, match="input byte budget"):
        Inputs(source, original, 1, False, lambda: None).load("anchor.pt")
    with pytest.raises(MemoryError, match="no output written"):
        write_outputs(tmp_path / "too-large", report, witness, maximum=100)
    assert not (tmp_path / "too-large").exists()
    with pytest.raises(ValueError, match="outside"):
        run(source, source / "new-output")
    with pytest.raises(MemoryError, match="RSS high-water"):
        run(source, tmp_path / "rss-limit", maximum_rss_gib=1e-8)
    assert not (tmp_path / "rss-limit").exists()
