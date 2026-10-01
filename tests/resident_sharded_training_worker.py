"""Fresh public consumer process: portable checkpoint, new owner count/placement."""
import argparse
import torch
from dataclasses import replace
from tidegraph import ResidentPlacement
from resident_half_training_cases import runtime, limits, update


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument("--cards", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    r = runtime("pdg", args.device, memory="attention", mode="softp", model_device="cpu")
    first = torch.device(args.device).index
    placement = ResidentPlacement(devices=tuple(f"npu:{first+i}" for i in range(args.cards)), policy="memory")
    values = (torch.arange(16).reshape(1, 4, 4)*.005).half()
    with torch.no_grad(), r.training_session(1, checkpoint=args.checkpoint, placement=placement, limits=replace(limits(), backward_bytes=8*1024**3)) as s:
        update(s, values, 2, 4)
        s.save(args.output)


if __name__ == "__main__":
    main()
