# Standalone C++ native value checkpoint

Clean source: `4325bb14434bbe0e9702aff244f77ed71e75cbee` (`4325bb1`). The
frozen detached worktree was clean and read-only. CPU aarch64 Linux used
`/home/zlong/anaconda3/bin/python`, Python 3.11.15 and Torch/LibTorch
2.10.0+cpu with the C++11 ABI. The build used two workers, one ATen/BLAS
thread, `TORCH_DEVICE_BACKEND_AUTOLOAD=0` and `Nice=10`.

The durable unit
`tide-foundation-native-checkpoint-qualified-20260922-121524` ran in
`background.slice` and terminated with `Result=success`, exit 0. Its durable
record and log are [status.json](../../artifacts/native-checkpoint-qualified-20260922-121524/status.json)
and [task.log](../../artifacts/native-checkpoint-qualified-20260922-121524/task.log).
The independent build is retained at
`/var/tmp/zlong-graph-execution-foundation/qualification/native-checkpoint-build-4325bb1`.

```sh
build=/var/tmp/zlong-graph-execution-foundation/qualification/native-checkpoint-build-4325bb1
python scripts/build.py --build-dir "$build" --jobs 2
"$build"/tidegraph-checkpoint-check --device=cpu --dtype=float64
"$build"/tidegraph-checkpoint-check --device=cpu --dtype=float32
"$build"/tidegraph-optimizer-check --device=cpu --dtype=float64
"$build"/tidegraph-optimizer-check --device=cpu --dtype=float32
PYTHONPATH="$PWD/python:$build" TIDE_BUILD_DIR="$build" \
  python -m pytest tests/test_cpp_checkpoint.py tests/test_cpp_optimizer.py \
    tests/test_checkpoint_ownership.py tests/test_single_graph_optimizer.py \
    -q --dtype both
```

The standalone checkpoint executable passed for FP64 and FP32. It exercises
`TIDENCK1` schema/identity, shared alias partitions, SGD and AdamW state,
undefined versus connected-zero gradients, next-update continuation,
identity/topology rejection, checksum/truncation preflight, unchanged-on-
failure and exclusive no-overwrite publication. The existing optimizer
executable passed for both dtypes. The directed Python gate passed **344 tests
in 61.95 seconds** and includes the new Python binding round trips plus the
existing native optimizer, ownership and single-graph optimizer suites.

Pytest emitted two cache permission warnings because the qualification
worktree was read-only; no test failed, and the worktree remained clean. The
build manifest records matching source and hashes for all ten built targets.
This gate covers named values and built-in optimizer state only. It does not
claim Python checkpoint-file interoperability, graph continuation/RNG restore,
SettleGraph construction from C++, CUDA/NPU support, full CPU regression or
M8 performance.
