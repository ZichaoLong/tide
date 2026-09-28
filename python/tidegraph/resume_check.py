"""Internal fresh-process worker for the public qualification gate."""
import sys
import torch
from .config import GraphConfig
from .library import GraphRuntime
from .qualification_inputs import Probe
from .qualification_training import trajectory


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    record = torch.load(sys.argv[1], weights_only=True, map_location="cpu")
    runtime = GraphRuntime(GraphConfig.from_dict(record["config"]), device="cpu", native_library=record["native_library"])
    results = trajectory(runtime, Probe.restore(record["probe"]), record["steps"], record["optimizer"], resume=record["checkpoint"])
    torch.save(results, sys.argv[2])


if __name__ == "__main__":
    main()
