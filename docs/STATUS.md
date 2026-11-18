# Current handoff — five-workstream implementation in progress

Branch: `graph-execution-foundation`. Starting HEAD: `f0452b822414f5cbce841ffc4ed7367e5865d70a`.
Fixed math: `4f5b9f20a4030d68ee66ae2d26e34659a5e37234`; all 108 files verified on 2026-11-18.
Scope remains the continuing FC/HB instance route; the general foundation is not active.

The current conversation authorizes the five ROADMAP workstreams, private builds,
finite CPU/NPU experiments, coordinated shared NPU use, staged commits and ordinary
pushes. Shared authorization expires with this conversation. Four pre-existing
untracked user documents remain protected and excluded from commits.

## Active implementation

- Joint A/B/C/D Transformer matching and independently checked parameter/work audit.
- Explicit FP32 parameters/gradients/AdamW with BF16 matrix arithmetic; sensitive
  state/reductions stay FP32. This policy is being implemented, not yet qualified.
- Real incremental Transformer KV cache and continuous Tide timing/restore checks.
- Chronological KV compaction and bounded Full packing/projection investigations.
- Fixed-source capacity planning, device qualification and finite measurements next.

No new qualification is claimed yet. Preserve the original four NPU FP32 finite
failures and restored AdamW deviations in [the original report](evidence/instances-fp32-20261118.md).
The governance/cleanup delivery remains in [its report](evidence/governance-cleanup-20261118.md).

## Resource preflight

2026-11-18: account quota about 38 GiB used / 300 GiB soft / 700 GiB hard (no grace
pressure), filesystem about 1.3 TiB free, host about 1.8 TiB available. Multiple NPUs
have no processes; occupied devices and other projects are left alone. No active
foundation experiment was found. A project-private `.venv-instances` uses the
published read-only `ascend/dev-workspace-9.2.0` baseline. Recheck resources before
large jobs. Durable runs live under ignored `artifacts/`, with immutable source,
queue/lifecycle records, explicit affinity and sampled process-tree RSS limits.

Implementation source must be committed before formal qualification, and reviewed
results committed separately. ROADMAP remains the five-workstream backlog; this
handoff will be replaced with concrete source/evidence identities as work completes.
