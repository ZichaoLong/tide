# Current handoff — governance complete, final cleanup

Branch: `graph-execution-foundation`. Governance/preservation source:
`9efc86e2aad143434ce526b021f41927f5c8eb50`.
Fixed mathematics: `4f5b9f20a4030d68ee66ae2d26e34659a5e37234` (108 files checked).
Active model: `c17c613f178dd076d963da98017abdea8c09db16`, unchanged.

[Project scope](project-scope.md) is the enduring agreement. [ROADMAP](ROADMAP.md)
is the five-workstream implementation programme to start in the user's new thread;
this governance task does not start BF16 or performance experiments.

## Completed and verified

Current navigation, contracts and three repository skills are aligned. Superseded
cycles leave the working tree; [history](history.md) locates explicitly requested
historical material. Model/test/benchmark/config/example/math source and original
FP32 evidence are unchanged. Links, skills and a fresh-context read were checked.

43 project roots and 131 experimental worktrees were retired after preserving
unique changes and small records at `9efc86e2`, under `retirement-preservation/`.
That bundle now exists only in Git. Inventoried allocations removed: 154.17 GiB disk
and 127.51 GiB tmpfs, plus small project caches. Four governance services are
terminal, inactive, MainPID=0; no experiment is live or queued.

A clean non-editable installation ran the existing CPU training/checkpoint example.
Restored forward was exact; the next real AdamW update had a recorded finite
2.771615982055664e-5 parameter deviation. This is not a numerical pass. Existing
FP32 CPU/NPU evidence and limits remain in [the FP32 report](evidence/instances-fp32-20261118.md).
BF16 and the four joint Transformer comparisons remain planned.
[Governance/cleanup evidence](evidence/governance-cleanup-20261118.md) owns this round's records.

## Protected items and capacity

Four user untracked documents remain untouched: TL-REQ-004/005/006 handoffs and
`upstream-lh-semantics-discussion-20261113.md`. The downstream worktree, its six
`gef-lm-experiments/20261102/models` aliases and tiny provenance files remain.
Reference repositories and public toolchains/models were not modified.

The old private environment was deliberately removed. Recreate it using
[development](development.md); no retired run directory is required.
Account quota is still about 352 GiB used / 300 GiB soft / 700 GiB hard, with
three-day grace observed. Remaining usage is outside this project; check quota
again before new large writes. The repository itself is about 113.3 MiB including Git.

## Immediate handoff

Commit the reviewed cleanup evidence, then retire the terminal 97.1 MiB control
directory `/mi/data2T/zlong/home/llm/gef-governance-20261118`, recheck protected items,
and publish the final accounting. No long job remains to resume.
After delivery, the user starts ROADMAP stage 1 through [the new-thread prompt](next-thread.md).
