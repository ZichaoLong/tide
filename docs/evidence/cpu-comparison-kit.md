# Portable LH / PDG CPU comparison kit

Qualified 2026-09-23 (Asia/Shanghai), implementation source
`dd024e6f1d57153c22ab7cef2762d059cbd3ac7d`. This increment packages the existing
measured implementations; it changes no model kernel or graph semantics.
See the [commands and configuration](../../tools/cpu_compare/README.md) and
[attention grouping explanation](../attention-grouping-comparison.md).

## Delivered scope

`run_lh.py` and `run_pdg.py` each verify the packet, prepare separate source,
configure/build against the target LibTorch, run the native process, and report
configuration, actual thread pools, per-token latency/work, warm-window
statistics, throughput, matrix FLOPs, padding and whole-process peak RSS.
Logs and run/summary/metrics/host/prepared-source records are retained on both
success and failure. Output directories must be new; timeout and interruption
reap the native process group. All computation remains C++/LibTorch.

Defaults reproduce the previous wide workload:17,269,426,339 parameters,
D2048/B512/V50304, four-head Attention/all-softmax, FP32/no_grad,12 tokens,
warmup4, seed7, fixed external IDs, two body ticks/token and clear.
Weights are independently initialized. Four CSR graph blocks are identical;
this is a comparable-compute workload, not an exact whole-model equality claim.
The prior large timing is scoped to [its own source](lh-pdg-operator-work.md).
No new large performance measurement was needed for this packaging increment.

The source-only archive includes both C++ implementations, the fixed graph and
vendored JSON headers/license. It requires no source checkout, graph generator,
Python LH interpreter, saved weights or source-machine input paths. Matching
Python Torch can locate LibTorch, or an explicit `--torch-prefix` can be used
without importing Python Torch. No package installation is performed.

## Qualification results

- Directed development helper/related-record tests:18 passed/0.98s; clean frozen
  repeat:18 passed/1.01s. Seven new tests cover live timer/partial JSONL ingestion,
  invalid metrics/counts, relocation/hash rejection, timeout process cleanup,
  help and explicit unsupported backend rejection.
- Exported the clean source, extracted into a path containing spaces, removed
  the redundant export staging directory, and verified all243 packet files.
- Fresh builds and six-token FP32 runs passed for LH4 threads, PDG4 workers and
  LH1 thread. Each used D16/B4/V257,1,059,267 parameters, warmup2.
- Both LH runs match the previous qualified six-token full logits exactly
  (6×1028 values each; maximum absolute error0). All model/work/operator counts
  also agree with the matching1/4-thread fixtures.
- PDG `--check 1` compares serial unpacked slot Emit against serial row Emit,
  serial packed and parallel packed paths with the current optimizations and
  counters. It checks complete trace, states, history, pending, routes and
  outputs. The fresh binary also agrees with the prior qualified binary on
  all six model/work/operator inventories and logit checksums (maximum error0).
  Checksum agreement is not a cross-model full-state equivalence claim.
- The third build used `/usr/bin/python3 -S` and an explicit LibTorch CMake
  prefix. The launcher could not import site packages. This verifies discovery
  without Python Torch; the SDK files came from the local Torch wheel, not a
  separately downloaded LibTorch archive.
- The absent-prefix negative control exits1, emits no observations, and retains
  `status=failed`. Its gate passed because rejection was expected; the failed
  run has not been relabeled as successful. All four portable records validate.
- Terminal audit verifies all424 frozen tracked files, packet hashes, prepared
  source hashes, binaries/CMake caches, archive checksum and complete statuses.

Environment: aarch64 CPU, Torch/LibTorch2.10.0+cpu, GCC10.3.1, C++11 ABI,
Python3.11.15; system Python3.9.9 with `-S` for explicit-prefix discovery.
LH4 actual ATen/OpenMP/OpenBLAS4, LH1 all1; original LH inter-op default320
(no inter-op task launches). PDG node/head workers4 in separate phases,
ATen/OpenMP/OpenBLAS/inter-op1. Target runners inherit affinity, use no extra
memory bound by default, and allow thread/build/timeout overrides.

The detached unit `tide-cpu-kit-20260923-0345` completed with exit0;
all12 driver stages passed, including the expected rejection. Persistent
status/pipeline agree; terminal unit was inactive/dead, MainPID0, Result=success.
While live it was verified outside focus.service in background.slice, Nice10,
CPUs160–319, at most4 build/native compute workers, RuntimeMaxSec3600.
The observed scope is clean relocated aarch64 FP32 build/smoke/parity. Intel
x86_64, other LibTorch versions and large runs of the packaged entry points
remain target-machine work. This is not another complete CPU regression gate.

## Retained artifacts

- Frozen source: `/var/tmp/zlong-graph-execution-foundation/qualification/cpu-kit-20260923-0345`.
- Driver: `artifacts/cpu-kit-runner-20260923-0345.py`.
- Records: `artifacts/cpu-kit-20260923-0345/`: status.json, pipeline.json,
  post-run-audit.json, task/stage logs, lh-smoke/, pdg-smoke/, lh-standalone/,
  pdg-old-native/, bad-prefix/, export/ and relocated kit with spaces/.
- Final archive: `artifacts/cpu-kit-20260923-0345/export/cpu-attention-compare.tar.gz`,
  382946 bytes; SHA256
  `26018705eddc387415dde5a05348e5756e8c1074dec3bf8e18d8730171c54ed9`.
  Manifest SHA256: `667719575d476d99559ace55a059b7ee76fc90063d97ded539a00b63ed0812f7`.
- Earlier wrong-interpreter failure remains in
  `artifacts/cpu-kit-dev-20260923-0340/initial-python-repro.json`; the shell's
  incomplete Torch namespace was not used for qualification. The dirty
  preview under cpu-kit-preview-20260923-0340 is not the delivered archive.

Trackio was best-effort/degraded (unavailable); all local records are complete.
No reference repository was modified; no push. The archive is the qualified
export itself, without a subsequent rebuild or provenance-changing repack.
