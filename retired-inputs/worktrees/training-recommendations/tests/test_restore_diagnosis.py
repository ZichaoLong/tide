import torch

from scripts.diagnose_restore import adamw_scalar, run_case
from tide import ModelConfig, TopologyConfig


def test_scalar_adamw_witness_matches_actual_fp64_update():
    parameter = torch.nn.Parameter(torch.tensor([.2], dtype=torch.float64))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4, foreach=False)
    optimizer.state[parameter] = {"step": torch.tensor(3.), "exp_avg": torch.tensor([1e-9], dtype=torch.float64),
                                  "exp_avg_sq": torch.tensor([1e-18], dtype=torch.float64)}
    parameter.grad = torch.tensor([.7e-9], dtype=torch.float64)
    expected = adamw_scalar(.2, .7e-9, 1e-9, 1e-18, 3, lr=1e-4, betas=(.9, .999), eps=1e-8, weight_decay=.01)
    optimizer.step()
    assert abs(float(parameter.detach()[0])-expected["parameter"]) < 1e-15


def test_restore_witness_separates_data_backward_and_update(tmp_path):
    cfg = ModelConfig(topology=TopologyConfig(n=8, layers=4), d=64, sharing="region_shared")
    record = run_case(cfg, "small-check", device=torch.device("cpu"), precision="bf16", batch=1, length=2,
                      steps=1, checkpoint=tmp_path / "checkpoint.pt")
    assert record["saved_data"]["exact"]
    assert record["all_forward_state_repeats_exact"]
    assert set(record["backwards"]) == {"same_original_owner", "same_restored_owner", "first_cross_owner", "second_cross_owner"}
    assert record["state"] in {"within_threshold", "finite_deviation"}
    assert not record["actual_updated_parameters"]["nonfinite"]
