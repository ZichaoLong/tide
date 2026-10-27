# Core-5 adoption and general SettleGraph — 2026-10-27

Implementation and tests: `efb9d629438fb77582842bb133158be7f9f70afb`.
Mathematics: `0770593838a37c790a568226930567ed60f99b83`, core-5 Settle §2–6,
§9–11; [source lock](../upstream.json), [mapping](../settle-embedding.md),
[semantic index](../semantic-map.md). The 19 new source files and five explicitly
legacy core-3 files passed byte/hash comparison against their pinned Git objects.

**247 passed, zero failed/skipped**, CPU FP64/FP32, clean immutable implementation
checkout. This is the directed scope below, not a full repository/backend matrix
or qualification of weighted KV, other new memories, allocations or devices.

## Reproduce from clean source

Use a matching project-private Python/Torch environment as in
[development](../development.md); no old run directory is required.
Run the following workload under the bounded durable-service procedure in
[verification](../verification.md):

```sh
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python scripts/build.py --backend cpu --build-dir build/core5-cpu --jobs 2
TIDE_BUILD_DIR="$PWD/build/core5-cpu" TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q --dtype both tests/test_semantics.py tests/test_native_settle.py tests/test_settle.py tests/test_settle_ports.py tests/test_upstream_lock.py tests/test_frontier.py::test_frontier_trace_vjp
python scripts/check_upstream.py
```

The actual qualification used `scripts/develop.py --reuse-build` with exactly
this test selection, after matching the C++ source hash and every binary hash.
The binaries had been built from the archived, frozen development tree
`4ded7416941fc279b04fad43242b544ee3e6fc51`; their C++ bytes match the subsequent
implementation commit. Tests ran only after that commit was fixed. This does not
relabel the build's recorded baseline/dirty-tree provenance as a clean build.

## What the tests establish

- `test_named_slots_literal_parallel_paths`: independent literal path coefficients
  (including two parallel edges), same-rank independent regions, distinct output
  ports, interior/ragged inputs, zero versus absent, and input-root derivatives.
- `test_every_cut_records_independent_roots_and_continuation`: every logical cut,
  full records, pending, states/history/control, port occurrence ledgers, independent
  output/state/pending roots and None/zero gradients across Python time/region
  schedules and native streaming/frontier/greedy. Shared parameters retain identity.
- `test_shared_optimizer_and_checkpoint_with_pending`: three AdamW steps, shared
  weights/private states, exact mid-position pending checkpoint and resumed update.
- `test_partial_full_slots_zero_messages_and_empty_selection`: sparse Full, present
  zero edge/output, absent parallel edge, unselected state update and empty selection.
- `test_time_code_validation_and_sparse_int64`: invalid timing/ranks/overflow fail;
  sparse time above 2^54 is processed exactly without scanning the time gap.
- `cpp/test/settle_ports.cpp`: standalone C++ literal coefficients and VJPs,
  source/port identities, every cut and rejection, without Python scheduling.
  The existing standalone recurrence and legacy broadcast/sum tests also pass.
- `test_semantics.py` retains hand-computed PDG feedback; the selected frontier
  cases retain nonaligned DAG paths, shared regions, complete traces and VJPs.

Common float tolerances: FP64 atol=1e-10/rtol=1e-8; FP32 1e-6/1e-5.
Discrete identities, activity, times, slots and gradient presence compare exactly.
The HST cases test its existing declared VJP, not a weighted allocation derivative.

## Environment, lifecycle and limits

Aarch64 HCE 2.0/glibc 2.34, GCC 10.3.1/C++17, Torch 2.12.0+cpu,
Python 3.12.13, CXX11 ABI enabled; project-private pytest 9.1.1/Ninja 1.13.2.
Public module `ascend/dev-workspace-9.1.0` was the environment entry; this run used
CPU and did not import or certify NPU execution. CUDA build/device claims are unchanged.

The new-toolchain baseline at `35d6358` built and passed 93 directed checks.
Its first attempt failed during CMake configuration because Ninja was missing;
`artifacts/jobs/core5-bootstrap` remains failed. Installing Ninja only in the private
venv repaired that prerequisite; `core5-bootstrap-tools` records the successful run.

Qualification service `tide-core5-settle-qualify-20261027` ran in `background.slice`,
Nice=10, two build workers, one ATen/BLAS thread, 30-minute cap; terminal inactive,
Result=success, workload exit 0. Raw logs/status/source archive are ignored under
`artifacts/jobs/core5-settle-qualify`; durable identity and selected hashes are in
[the small report](core5-settle-20261027.json). No performance or resource isolation
claim is inferred. Weighted local records, remaining candidates, mixed/resident and
GPU hardware validation remain separate work items in ROADMAP.
