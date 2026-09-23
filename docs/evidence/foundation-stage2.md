# Stage 2 graph interfaces and independent schedules

Frozen implementation: `70809fd1acb117ca39881abc2b90b59c7cc2ddb0`. Clean read-only worktree
`/var/tmp/zlong-graph-execution-foundation/qualification/foundation-stage2-70809fd`.
Standalone native SettleGraph construction/encoding, no-Python loader closure and
literal formula/VJP gate remain in [native frontend](native-settle-frontend.md).

Independent Python/C++ ring and diamond schedules now complement self-loop and
chain. Diamond includes shared middle-region competition; Python layered Settle
uses its own scalar region/layer schedule. Full trace, state slots, histories,
source/edge/port identity, messages/pending and occurrence ledgers align with the
generic engines in representative cases. Tests include delay/ragged/empty/clear,
HST, shared/unused parameter and initial state roots, and every-cut continuation.
Settle encoding/projection is restricted as declared in settle-embedding.md;
this is no proof for arbitrary modules or graphs.

Directed dirty-source gate:554 passed/61.54s at
`artifacts/specialized-topologies-dev-20260923-a/`. Original redundant-squared
loss FP32 failure is retained at `artifacts/specialized-linear-fp32-repro-20260923/`;
direct objective tests passed unchanged tolerances, no kernel workaround.

Clean complete CPU FP64/FP32 gate: **7167 passed in933.90s**, build and tests exit0.
Exact commands, source inventory, build/binary fingerprints and terminal status:
`artifacts/foundation-stage2-20260923-a/` (`pipeline.json`, `full-cpu/result.json`,
`full-cpu/tests.log`, `reviewed-audit.json`, `status.json`). Build2, ATen/BLAS1;
aarch64, Torch2.10.0+cpu, GCC10.3.1, Python3.11.15, C++11 ABI. Detached unit
`tide-foundation-stage2-20260923-a`, background.slice/Nice10, RuntimeMaxSec3600.
It overlapped only a small correctness build/test gate, no formal timing.

Audit checked 459 source hashes against Git archive and the still-clean
snapshot, every binary against its manifest, all actual exits and absence of
remaining cgroup processes. Unit inactive/dead, MainPID0/ExecMainStatus0.
Native adapter SHA256 `175c3adb90e9f2c61659af95bc1852a9a46d24dbfd0c19dc58f0993304d92659`.

S2 is accepted. This source predates S3 frontier transport migration, ungated
DeltaRule and model adapter; no stage3 or performance qualification is implied.
