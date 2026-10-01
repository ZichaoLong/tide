# Update-scoped projection snapshot qualification

Implementation `3f85852552c607e7e44dc9480d41eafca1027081`; immutable clean source.
[Machine-readable audit](resident-projection-retention-20261002.json).

Single-device and sharded public training now retain one independent immutable
projection-bank copy per update. Later retained windows reuse it; backward,
detach and close release it before parameter publication. Other parameter banks,
state, KV and journals retain their existing ownership. The generic two-argument
internal retain APIs still make independent copies. CANN writes need not increment
ATen versions: the public owner's update lifecycle is part of this guarantee.

Eight qualification jobs passed: standalone/Python-owned backends and installed
client builds; 26 library cases; 32 FP32/FP16 sharded trajectories and 12 single-card
FP16 trajectories (704 windows,176 updates); 27 actual-consumer checks; separate
profiling. No skips, tolerance changes or fallback warnings were accepted.

The new three-family cases compare full observables, gradients, three nonzero
AdamW updates and save/restore against independent CPU autograd. They run with the
exact two-window reservation, reject a second window with one byte less before
changing the cut/state, then successfully continue after explicit detach.

| Case | Per-window bytes | Shared projection bytes | Two-window bytes |
| --- | ---: | ---: | ---: |
| PDG, one card | 187828 | 720 | 376376 |
| TimedDAG, two cards | 188188 | 640 | 377016 |
| SettleGraph, two cards | 188720 | 560 | 378000 |

These are small semantic boundary cases, not large-model memory measurements.
Statistics report retained tensor reservations, not allocator peaks. The saved
bytes for W retained windows are (W-1) times the physical projection-bank bytes.

Actual consumers cover both runtime owners, three families, both schedules,
Add/Attention,12 inference and12 complete-training trajectories,96 windows and24
updates. The independent D32/two-card Attention trace includes four continued
windows/two AdamW updates:14782 AI_VECTOR_CORE,607 AI_CORE,218 MIX_AIV tasks and
no observed AiCPU. It includes construction and instrumentation; no throughput
speedup is inferred. Full matrices and parameter partials still live on the
coordinator; compact projection sharding and total-memory admission remain required.

Build qualification reuses byte-verified unchanged core/device artifacts and
identical affected development objects. The installed client reuses nine objects
only after matching client sources, installed public headers and normalized compiler
commands, then freshly links and verifies loader closure to the new backend. This
is an audited incremental build, not a from-scratch recompilation claim.
