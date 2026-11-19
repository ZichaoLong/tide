"""Benchmark boundaries must retain optimizer updates and real prior roots."""
import torch

from benchmarks.reception import evaluate, make_fixture
from benchmarks.continuous import nested_comparison
from scripts.qualify_precision import fingerprint


def test_reception_benchmark_training_masks_absent_roots_and_performs_adamw():
    fixture, spec = make_fixture("incoming_light_64", "padded", nodes=3, batch=1,
                                 length=2, edges=3, d=64, sharing="shared")
    observations = []
    for candidate in ("separate", "fused_masked"):
        report, snapshot = evaluate(fixture, spec, device=torch.device("cpu"), precision="fp32",
                                    mode="training", candidate=candidate, backend="math", recompute=True,
                                    warmup=1, steps=2)
        assert report["state"] == "completed"
        assert len(report["seconds"]) == 2
        assert report["training_updates_including_warmup_and_audit"] == 4
        assert {int(state["step"]) for state in snapshot["optimizer"]["state"].values()} == {4}
        padded = (~fixture["membership"])[:, None, None].expand_as(snapshot["gradients"]["raw_mass"])
        assert not snapshot["gradients"]["raw_mass"][padded].count_nonzero()
        assert any(not torch.equal(snapshot["parameters"][name], value) for name, value in fixture["parameters"].items())
        assert report["audit"]["attention"]["effective"] == "math"
        assert len(report["snapshot_sha256"]) == 64
        observations.append(snapshot)
    assert nested_comparison(*observations)["within_threshold"]


def test_structural_mask_changes_dispatch_but_real_zero_priors_keep_tiled_vjp():
    for case, expected in (("padded", "sdpa"), ("zero", "tiled")):
        fixture, spec = make_fixture("incoming_light_128", case, nodes=3, batch=1,
                                     length=2, edges=3, d=64, sharing="shared")
        report, _ = evaluate(fixture, spec, device=torch.device("cpu"), precision="fp32",
                             mode="forward_backward", candidate="masked", backend="auto", warmup=0, steps=1)
        assert report["audit"]["attention"]["effective"] == expected


def test_fingerprint_accepts_scalar_optimizer_counters_without_collapsing_shape():
    scalar = torch.tensor(2.)
    assert fingerprint(scalar) == fingerprint(scalar.clone())
    assert fingerprint(scalar) != fingerprint(scalar+1)
    assert fingerprint(scalar) != fingerprint(scalar[None])
