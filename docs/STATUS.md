# Current handoff — governance and project cleanup complete

Branch: `graph-execution-foundation`. Governance/preservation source:
`9efc86e2aad143434ce526b021f41927f5c8eb50`.
Fixed mathematics: `4f5b9f20a4030d68ee66ae2d26e34659a5e37234` (108 files checked).
Active model: `c17c613f178dd076d963da98017abdea8c09db16`, unchanged.

[Project scope](project-scope.md) is the enduring agreement. [ROADMAP](ROADMAP.md)
is the five-workstream implementation programme to start in the user's new thread.
This governance task has not started BF16 or performance implementation/experiments.

## Completed and verified

Current navigation, contracts and three repository skills are aligned. Superseded
cycles leave the working tree; [history](history.md) locates explicitly requested
historical material. Model/test/benchmark/config/example/math source and original
FP32 evidence are unchanged. Links, skills and a fresh-context read were checked.

43 project roots and 131 experimental worktrees were retired after preserving
unique changes and small records at `9efc86e2`, under `retirement-preservation/`.
That bundle now exists only in Git. Inventoried allocations removed: 154.17 GiB disk
and 127.51 GiB tmpfs, plus small project caches and the 97.1 MiB validation/control
directory. Four governance services are terminal, inactive, MainPID=0. No foundation
job is live or queued. Separate downstream jobs were left untouched.
After the user's confirmation that the old inputs have no dependent consumers,
[supplementary cleanup](evidence/cleanup-followup-20261118.md) removed two historical
source copies, their wheel caches, the old model aliases/provenance and ten empty
temporary trees: another 93.875 MiB across 15 explicit roots.

A clean non-editable installation ran the existing CPU training/checkpoint example.
Restored forward was exact; the next real AdamW update had a recorded finite
2.771615982055664e-5 parameter deviation. This is not a numerical pass. Existing
FP32 CPU/NPU evidence and limits remain in [the FP32 report](evidence/instances-fp32-20261118.md).
BF16 and the four joint Transformer comparisons remain planned.
[Governance/cleanup evidence](evidence/governance-cleanup-20261118.md) owns this round's records.

## Protected items and capacity

Four user untracked documents remain untouched: TL-REQ-004/005/006 handoffs and
`upstream-lh-semantics-discussion-20261113.md`. The downstream worktree and its
Git association remain; both worktrees still share this repository's `.git`.
The old `gef-lm-experiments` aliases/provenance directory has been removed;
public model targets, reference repositories and public toolchains were not modified.

The old private environment was deliberately removed. Recreate it using
[development](development.md); no retired run directory is required.
Account quota is still about 352 GiB used / 300 GiB soft / 700 GiB hard, with
three-day grace observed. Remaining usage is outside this project; check quota
again before new large writes. The repository is about 114 MiB including Git,
with about 3.5 MiB working content and no ignored artifacts left.

## Next action

The user starts ROADMAP stage 1 through [the new-thread prompt](next-thread.md).
No old job, queue, environment or run directory needs to be restored. Begin with:

```sh
git status --short --branch
python scripts/status.py
python scripts/check_upstream.py
```

Then follow AGENTS, project-scope and ROADMAP. Current implementation is the active
starting point; retired APIs and backend matrices do not impose new requirements.
