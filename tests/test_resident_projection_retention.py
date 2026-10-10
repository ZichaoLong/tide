"""One frozen projection version per update, with exact retained-capacity gates."""
import json
import pytest
import torch
from resident_test_target import owner_devices
from tidegraph import ResidentPlacement, ResidentTrainingLimits
from tidegraph.compare import equivalent
from test_resident_training import target, training_case
from resident_training_cases import runtime, inputs


@pytest.mark.parametrize("family,cards,schedule", [
    ("pdg", 1, "streaming"), ("timed-dag", 2, "greedy"), ("settle", 2, "streaming")])
def test_projection_snapshot_budget_and_update_lifetime(target, family, cards, schedule, tmp_path):
    start = torch.device(target).index
    placement = (ResidentPlacement(devices=owner_devices(target, cards)) if cards>1 else None)
    options = dict(full="lh-silu-rms-v1", aggregation="all_softmax", emission="slot_affine",
                   placement=placement, model_device="cpu", root_modes=("all", "all", "all"))
    broad, bounded = tmp_path/"broad", tmp_path/"bounded"
    broad.mkdir(); bounded.mkdir()
    # Each run independently checks CPU autograd, complete state, aliases,
    # three nonzero updates and a saved/restored optimizer suffix. The later
    # gradients must see newly published weights, not the previous snapshot.
    first = training_case(target, family, schedule, "adamw", broad, **options)
    peak = first[0]["retained_bytes"]
    for s in first:
        assert s["retained_windows"] == 2 and s["retained_projection_bytes"] > 0 and s["retained_full_bytes"] > 0
        assert s["retained_bytes"] == 2*s["retained_window_bytes"]+s["retained_projection_bytes"]+s["retained_full_bytes"] == peak
        assert peak < 2*(s["retained_window_bytes"]+s["retained_projection_bytes"]+s["retained_full_bytes"])
    limits = ResidentTrainingLimits(retained_bytes=peak, backward_bytes=8*1024**3 if placement else 512*1024**2)
    second = training_case(target, family, schedule, "adamw", bounded, training_limits=limits, **options)
    assert first == second
    (tmp_path/"retention.json").write_text(json.dumps(dict(family=family, cards=cards, statistics=second)))

    r = runtime(family, target, schedule, full="lh-silu-rms-v1", aggregation="all_softmax",
        emission="slot_affine", model_device="cpu", resident_workspace_bytes=(1024**3 if placement else 64*1024**2))
    one_short = ResidentTrainingLimits(retained_bytes=peak-1, backward_bytes=limits.backward_bytes)
    values = torch.ones(2, 2, 4)*.01
    with torch.no_grad(), r.training_session(2, placement=placement, limits=one_short) as s:
        args, kw = inputs(s, values, 0, 1)
        s.advance_device(args, **kw)
        before, cut = s.result(), s.cut
        args, kw = inputs(s, values, 1, 2)
        with pytest.raises(ValueError, match="retained-window capacity"):
            s.advance_device(args, **kw)
        assert s.cut == cut and s.retained_windows == 1
        equivalent(before, s.result())
        # Capacity refusal happens before device progress and does not poison it.
        s.detach()
        s.advance_device(args, **kw)
        assert s.retained_windows == 1 and s.cut > cut
