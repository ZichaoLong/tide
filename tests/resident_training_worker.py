"""Fresh process uses the saved complete training state and its own device execution."""
import argparse
import torch
from resident_training_cases import runtime, inputs, roots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    r = runtime("pdg", args.device)
    values = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4) * .005
    with torch.no_grad(), r.training_session(1, checkpoint=args.checkpoint) as s:
        external, kw = inputs(s, values, 2, 4)
        w = s.advance_device(external, **kw)
        s.backward([roots(s, w, "all")])
        assert s.step().applied
        s.save(args.output)


if __name__ == "__main__":
    main()
