"""Public placed-node gradients, optimizers and relocated exact training cuts.

Each candidate consumes only its own windows. The shared test consumer uses
independent Python scheduling/autograd and FP32 optimizer algebra.
"""
import pytest
import torch
from test_resident_weighted_training import check_cut_updates_and_disk_resume


@pytest.mark.parametrize("family,topology,kind", [
    ("kv", "fc", "sgd"), ("kv", "hb", "adamw"),
    ("gated_delta", "dag", "sgd"), ("ssd", "pdg", "adamw")])
@pytest.mark.parametrize("storage_dtype", [torch.float32, torch.float16])
def test_placed_updates_complete_gradients_and_relocated_disk_resume(family, topology, kind, storage_dtype, tmp_path):
    check_cut_updates_and_disk_resume(family, topology, kind, storage_dtype, tmp_path, placed=True)
