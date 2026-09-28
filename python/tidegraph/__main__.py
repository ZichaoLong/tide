"""Installed-library CLI. No source-tree or benchmark-script dependency."""
import argparse
import json
from .config import GraphConfig


def main():
    parser = argparse.ArgumentParser(prog="tidegraph", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="normalize and identify a graph configuration")
    inspect.add_argument("config")
    qualify = commands.add_parser("qualify", help="finite CPU observable/VJP/trajectory/checkpoint gate")
    qualify.add_argument("config")
    qualify.add_argument("--device", required=True, choices=("cpu", "npu", "cuda", "auto"))
    qualify.add_argument("--output-dir", required=True)
    qualify.add_argument("--native-library")
    qualify.add_argument("--width", type=int)
    qualify.add_argument("--dtype", choices=("float32", "float64"))
    qualify.add_argument("--batch-size", type=int, default=2)
    qualify.add_argument("--positions", type=int, default=4)
    qualify.add_argument("--stop", type=int)
    qualify.add_argument("--seed", type=int, default=19, help="qualification input seed; model seed is in config")
    qualify.add_argument("--steps", type=int, default=3)
    qualify.add_argument("--optimizer", choices=("adamw", "sgd", "momentum"), default="adamw")
    qualify.add_argument("--threads", type=int, default=1)
    qualify.add_argument("--atol", type=float, help="explicit tensor tolerance, recorded in the report")
    qualify.add_argument("--rtol", type=float, help="explicit tensor tolerance, recorded in the report")
    args = parser.parse_args()
    try:
        config = GraphConfig.load(args.config)
        if args.command == "inspect":
            print(json.dumps(dict(config=config.to_dict(), sha256=config.identity), indent=2))
            return
        if args.threads < 1:
            raise ValueError("threads must be positive")
        import torch
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        from .qualification import qualify as run
        report = run(config, device=args.device, output_dir=args.output_dir, native_library=args.native_library,
                     width=args.width, dtype=args.dtype, batch_size=args.batch_size, positions=args.positions,
                     stop=args.stop, input_seed=args.seed, steps=args.steps, optimizer=args.optimizer,
                     atol=args.atol, rtol=args.rtol)
        print(json.dumps(dict(state=report["state"], effective_sha256=report["effective_sha256"],
                              coverage=report["coverage"], checks=report["checks"]), indent=2))
    except (ValueError, RuntimeError, AssertionError, OSError) as error:
        parser.exit(1, f"{type(error).__name__}: {error}\n")


if __name__ == "__main__":
    main()
