# Standalone C++ named ownership and optimizer parity

Clean source: `5c440e1648070c74eb25955893e10bffd3091d5c` (`5c440e1`). The
frozen worktree was clean and read-only. CPU aarch64 Linux used
`/home/zlong/anaconda3/bin/python`, Python 3.11.15 and Torch/LibTorch
2.10.0+cpu with the C++11 ABI. The build used two workers, one ATen/BLAS
thread and `TORCH_DEVICE_BACKEND_AUTOLOAD=0`.

The durable unit
`tide-foundation-named-optimizer-qualified-20260922-1010` ran in
`background.slice` with Nice=10. Its terminal record is
`artifacts/named-optimizer-qualified-20260922-1010/status.json`, and its
independent build is `/var/tmp/zlong-graph-execution-foundation/qualification/named-optimizer-build`.

```sh
build=/var/tmp/zlong-graph-execution-foundation/qualification/named-optimizer-build
python scripts/build.py --build-dir "$build" --jobs 2
"$build"/tidegraph-optimizer-check --device=cpu --dtype=float64
"$build"/tidegraph-optimizer-check --device=cpu --dtype=float32
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH="$PWD/python:$build" TIDE_BUILD_DIR="$build" \
  python -m pytest tests/test_cpp_optimizer.py \
    tests/test_checkpoint_ownership.py tests/test_single_graph_optimizer.py \
    -q --dtype both
```

Both standalone checks passed for FP64 and FP32. The directed Python gate
passed **338 tests in 64.18 seconds**. It compares native LibTorch SGD and
AdamW updates to PyTorch, including two and three step continuations, shared
cross-graph aliases, canonical owner/group order, subset groups, undefined
gradients, connected-zero gradients, momentum, decoupled weight decay,
AMSGrad and explicit AdamW epsilon. The executable also checks Model field
registration and duplicate-owner rejection. The existing Python ownership and
single-graph optimizer suites remained green.

The implementation is `cpp/include/tide/parameters.h` plus
`cpp/include/tide/optimizer.h` and their separate source files. It is a CPU
FP32/FP64 optimizer layer with no value codec, checkpoint schema or
transactional persistence. SettleGraph construction and encoding remain a
Python frontend; native execution still consumes the resulting encoded Graph.
CUDA/NPU and wider full-regression/performance claims are outside this gate.
