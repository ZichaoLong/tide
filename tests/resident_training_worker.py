"""Fresh process uses the saved complete training state and its own device execution."""
import argparse
import torch
from resident_training_cases import runtime as ordinary_runtime, inputs, roots as ordinary_roots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--full", default="tanh", choices=("tanh", "swiglu", "lh-silu-layer-v1"))
    parser.add_argument("--aggregation", default="sum", choices=("sum", "mean", "weighted_mean", "active_softmax", "all_softmax"))
    parser.add_argument("--mode", default="hard", choices=("hard", "hst", "softp"))
    parser.add_argument("--memory", choices=("ema", "event"), default="ema")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.memory == "event":
        from resident_event_cases import runtime, roots
        r = runtime("pdg", args.device, mode=args.mode)
    else:
        roots = ordinary_roots
        r = ordinary_runtime("pdg", args.device, full=args.full, aggregation=args.aggregation, mode=args.mode)
    values = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4) * .005
    with torch.no_grad(), r.training_session(1, checkpoint=args.checkpoint) as s:
        external, kw = inputs(s, values, 2, 4)
        w = s.advance_device(external, **kw)
        s.backward([roots(s, w, "all")])
        assert s.step().applied
        s.save(args.output)


if __name__ == "__main__":
    main()
