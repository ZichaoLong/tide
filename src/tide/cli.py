"""Command-line entry points for data preparation, probes, and training."""

from __future__ import annotations

import argparse
import sys
from typing import Any

from .runtime import (
    DTYPES,
    RuntimeConfigurationError,
    RuntimeRequest,
    RuntimeUnavailableError,
    normalize_device,
    resolve_runtime,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def _layer_indices(value: str) -> list[int]:
    try:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("layer indices must be comma-separated integers") from exc
    if not result or any(item < 0 for item in result):
        raise argparse.ArgumentTypeError("at least one nonnegative layer index is required")
    return result


def _add_runtime(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device",
        required=True,
        help="explicit runtime: cpu, cuda, npu, auto, cuda:N, or npu:N",
    )
    parser.add_argument("--device-index", type=_nonnegative_int, default=None)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="auto")
    parser.add_argument("--seed", type=_nonnegative_int, default=42)


def _runtime_from_args(args: argparse.Namespace):
    backend, index = normalize_device(args.device, args.device_index)
    return resolve_runtime(RuntimeRequest(backend, index, args.dtype, args.seed))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tide")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data", help="tokenize the pinned FineWeb-Edu split")
    prepare.add_argument("--model-path", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--validation-tokens", type=_positive_int, default=1_000_000)
    prepare.add_argument("--train-tokens", type=_positive_int, default=10_000_000)
    prepare.add_argument("--sequence-length", type=_positive_int, default=512)

    probe = subparsers.add_parser("probe", help="run receiver and Qwen3 forward/backward gates")
    _add_runtime(probe)
    probe.add_argument("--model-path", required=True)
    probe.add_argument("--output-dir", required=True)
    probe.add_argument("--profile", choices=["selected-dispatch", "bo"], default="bo")
    probe.add_argument("--layer-index", type=_nonnegative_int, default=13)
    probe.add_argument("--batch-size", type=_positive_int, default=1)
    probe.add_argument("--sequence-length", type=_positive_int, default=16)
    probe.add_argument("--attention-implementation", choices=["eager", "sdpa"], default="sdpa")

    train = subparsers.add_parser("train", help="run one D0, S1, or B1 experiment")
    _add_runtime(train)
    train.add_argument("--model-path", required=True)
    train.add_argument("--data-dir", required=True)
    train.add_argument("--output-dir", required=True)
    checkpoint = train.add_mutually_exclusive_group()
    checkpoint.add_argument("--init-from", default=None)
    checkpoint.add_argument("--resume", default=None)
    train.add_argument("--profile", choices=["d0", "selected-dispatch", "bo"], required=True)
    train.add_argument("--layer-indices", type=_layer_indices, default=[6, 13, 20, 27])
    train.add_argument("--receiver-count", type=_positive_int, default=4)
    train.add_argument("--state-size", type=_positive_int, default=128)
    train.add_argument(
        "--implementation",
        choices=["dense-masked-reference", "packed"],
        default="packed",
    )
    train.add_argument("--scan-implementation", choices=["reference", "vectorized"], default="vectorized")
    train.add_argument("--attention-implementation", choices=["eager", "sdpa"], default="sdpa")
    train.add_argument("--sequence-length", type=_positive_int, default=512)
    train.add_argument("--micro-batch-size", type=_positive_int, default=4)
    train.add_argument("--gradient-accumulation", type=_positive_int, default=4)
    train.add_argument("--evaluation-batch-size", type=_positive_int, default=4)
    train.add_argument("--max-tokens", type=_positive_int, default=10_000_000)
    train.add_argument("--max-steps", type=_positive_int, default=None)
    train.add_argument("--validation-tokens", type=_positive_int, default=1_000_000)
    train.add_argument("--backbone-lr", type=float, default=1e-5)
    train.add_argument("--extension-lr", type=float, default=1e-4)
    train.add_argument("--beta1", type=float, default=0.9)
    train.add_argument("--beta2", type=float, default=0.95)
    train.add_argument("--weight-decay", type=float, default=0.1)
    train.add_argument("--gradient-clip", type=float, default=1.0)
    train.add_argument("--balance-coefficient", type=float, default=0.01)
    train.add_argument("--warmup-ratio", type=float, default=0.05)
    train.add_argument("--minimum-lr-ratio", type=float, default=0.1)
    train.add_argument("--checkpoint-every", type=_nonnegative_int, default=100)
    train.add_argument("--log-every", type=_positive_int, default=1)
    train.add_argument(
        "--no-initial-validation",
        action="store_false",
        dest="run_initial_validation",
        help="skip the initial validation pass (useful only for smoke tests)",
    )
    train.set_defaults(run_initial_validation=True)
    return parser


def _as_arguments(args: argparse.Namespace) -> dict[str, Any]:
    values = vars(args).copy()
    values.pop("command", None)
    values["layer_indices"] = list(values.get("layer_indices", []))
    return values


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare-data":
            from .data import prepare_fineweb

            prepare_fineweb(
                model_path=args.model_path,
                output_dir=args.output_dir,
                validation_tokens=args.validation_tokens,
                train_tokens=args.train_tokens,
                sequence_length=args.sequence_length,
            )
            return 0

        runtime = _runtime_from_args(args)
        print(
            f"resolved runtime: {runtime.info.device}/{runtime.info.dtype} "
            f"({runtime.info.resolution_reason})",
            flush=True,
        )
        arguments = _as_arguments(args)
        if args.command == "probe":
            from .train import run_probe

            run_probe(runtime, arguments)
            return 0
        if args.command == "train":
            from .train import run_training

            run_training(runtime, arguments)
            return 0
        parser.error(f"unknown command: {args.command}")
    except (
        RuntimeConfigurationError,
        RuntimeUnavailableError,
        FileExistsError,
        FileNotFoundError,
        ValueError,
    ) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
