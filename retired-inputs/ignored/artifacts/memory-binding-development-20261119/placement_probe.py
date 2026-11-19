"""Bounded development witness, not committed-source/device qualification."""
import copy
import hashlib
from pathlib import Path
import runpy
import subprocess
import sys

import torch

from benchmarks.continuous import nested_comparison
from scripts.job import atomic
from tide import TideModel, TopologyConfig, canonical_state, canonical_state_dict, place_model, placement_adamw


def main():
    torch.set_num_threads(1)
    helpers = runpy.run_path("tests/test_memory_sharing.py")
    source = [*Path("src/tide").glob("*.py"), Path("tests/test_memory_sharing.py"), Path(__file__)]
    record = dict(kind="precommit_bounded_development_diagnostic", device="cpu", threads=1,
                  torch=torch.__version__, python=sys.version, seed=83,
                  head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source}, trials=[])
    for region_batch in (1, 2):
        for dtype in (torch.float32, torch.float64):
            torch.manual_seed(83)
            config = helpers["config"](
                topology=TopologyConfig(kind="multiscale", n=2, hubs=1, profile="compact-5-dd4"),
                region_batch_size=region_batch, checkpoint_regions=True, parameter_indexing="static_view")
            reference = TideModel(config).to(dtype=dtype)
            partitions = [(reference.graph.nodes[nodes[0]].position+reference.graph.nodes[nodes[0]].layer) % 2
                          for nodes in reference.graph.regions]
            placed = place_model(copy.deepcopy(reference), ["cpu:0", "cpu:1"], region_partitions=partitions)
            optimizers = [torch.optim.AdamW(reference.parameters(), lr=1e-4), placement_adamw(placed, lr=1e-4)]
            with torch.inference_mode():
                placed(torch.randn(1, 1, 64, dtype=dtype))
            states = [None, None]
            trial = dict(config=config.to_dict(), dtype=str(dtype), partitions=partitions, steps=[])
            for step in range(3):
                data = torch.randn(1, 2, 64, dtype=dtype)
                results = []
                for index, model in enumerate((reference, placed)):
                    result, states[index], dx, gradients = helpers["training_step"](
                        model, optimizers[index], data, states[index])
                    results.append((result, dx, gradients))
                tolerance = dict(atol=1e-6, rtol=2e-3) if dtype == torch.float32 else dict(atol=2e-9, rtol=2e-8)
                a, b = results
                observed = dict(value=nested_comparison(a[0].values, b[0].values, **tolerance),
                                strengths=nested_comparison(a[0].strengths, b[0].strengths, **tolerance),
                                output_support_equal=torch.equal(a[0].strengths > 0, b[0].strengths > 0),
                                input_gradient=nested_comparison(a[1], b[1], **tolerance),
                                parameter_gradient=nested_comparison(a[2], b[2], **tolerance),
                                state=nested_comparison(states[0], canonical_state(placed, states[1]), **tolerance),
                                parameters=nested_comparison(canonical_state_dict(reference),
                                                             canonical_state_dict(placed), **tolerance))
                trial["steps"].append(dict(update=step+1, comparisons=observed))
            record["trials"].append(trial)
    atomic(Path(__file__).with_suffix(".json"), record)


if __name__ == "__main__":
    main()
