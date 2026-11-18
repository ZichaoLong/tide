# Instance governance and artifact retirement — 2026-11-18

Governance source: `9efc86e2aad143434ce526b021f41927f5c8eb50`.
Active model source: `c17c613f178dd076d963da98017abdea8c09db16`.
Fixed mathematics: `4f5b9f20a4030d68ee66ae2d26e34659a5e37234` (108 files).
[Reviewed machine-readable record](governance-cleanup-20261118.json) contains
identities, manifests, terminal job facts, validation code and results.

## Scope and preservation

[Project scope](../project-scope.md) separates the continuing instance product
route from the separately activated general foundation. The five agreed next
workstreams, BF16 target and four joint Transformer matches have explicit owners
in ROADMAP and its contracts. Three repository skills link those owners. The
[new-thread prompt](../next-thread.md) is a template, not an automatic launch.

1,017 superseded tracked documents, evidence files and old snapshots leave the
active tree. Their original bytes remain at pre-governance commit
`728d19a470d16d392d5dd6a2452161d1a3e9959e`; normal navigation uses current material
and concise performance lessons. Model, tests, benchmarks, configurations, scripts,
examples, package metadata, mathematics and the two current FP32 evidence files
are byte-unchanged from that commit. This work adds no BF16 or performance result.

Before disk deletion, unique changes in 27 experimental worktrees and 145 untracked
files were preserved. The 52.18 MiB compressed preservation bundles and manifests
are committed at `9efc86e2aad143434ce526b021f41927f5c8eb50`, under
`retirement-preservation/`; they are removed from the final working tree.
Hashes, reachable Git blobs and unique source bytes were checked. Of 547,444
record entries, 41,954 unique files retain exact bundled bytes and 1,055 entries
are JSON summaries. Large raw results, including 170 JSON files above 4 MiB,
were deliberately not retained byte-for-byte. Other entries reference reachable
Git blobs, duplicates or retained log tails. This is not a promise of complete
historical raw-run recovery. See [history](../history.md) for explicit lookup.

## Disk cleanup and retained dependencies

| Cleanup | Observed amount |
|---|---:|
| Audited project roots removed | 43 |
| Registered experimental worktrees removed | 131 |
| Original allocated disk bytes in those roots | 165,543,432,192 (154.17 GiB) |
| Original allocated tmpfs bytes in those roots | 136,907,603,968 (127.51 GiB) |
| Additional project Python cache directories | 7 |
| Additional positively identified project wheel-cache leaves | 35 |

These are inventoried allocated bytes, not an account-wide free-space delta.
277 old failed transient service registrations were released after preserving
their terminal facts; none was relabelled passed. Four governance services
(preservation, curation, deletion, installed validation) terminated successfully
in `background.slice`; their individual results remain in the JSON record.

The main and `tide-learning-lab` worktrees remain. The downstream HEAD and clean
status, four user-owned untracked documents and six downstream model aliases
were verified unchanged. Tiny external experiment provenance files (2,351 bytes)
remain with those aliases. Reference repositories, downstream installations,
public software/models and unrelated caches were not cleaned. The old private
development environment was retired; [development](../development.md) rebuilds it.

After the main cleanup the repository occupied about 113.3 MiB including Git.
The 97.1 MiB temporary governance/validation directory is the last cleanup item;
its reviewed result is committed before that directory is removed.
Account quota changed from approximately 507 GiB to 352 GiB used, with a 300 GiB
soft limit, 700 GiB hard limit and three-day grace observed. Remaining usage is
outside the project cleanup. Check quota and grace before new large writes;
filesystem free space alone does not describe the account's writable capacity.

## Clean installation and numerical scope

A frozen export of source `9efc86e2` was installed non-editably into a fresh private
venv using the published `ascend/dev-workspace-9.2.0` module: Python 3.12.13,
Torch 2.12.0+cpu and installed TorchNPU distribution 2.12.0.post3. Vendor autoload
was disabled; this check did not execute an NPU workload. The installed package
resolved inside the new venv, every installed source file matched the export,
and all 196 original export files were unchanged by packaging.

The existing public training example ran three CPU AdamW updates (four threads),
saved/restored the checkpoint and optimizer, reproduced the next forward exactly,
and performed a real resumed update. That update reported a finite parameter
deviation of **2.771615982055664e-5** in `regional.1.second.bias`, beyond the
example's `atol=1e-6, rtol=0.002` criterion. Optimizer-state comparison was within
tolerance. The service completed and recorded the deviation; this is **not** a
claim that the resumed update passed its numerical threshold. Existing FP32
limits remain, and ROADMAP includes their bounded numerical investigation.

The 108-file math check and benchmark CLI help passed. Current local links and
three skill validators passed. An independent fresh-context review checked
28 pages / 139 links and recovered the intended scope without a blocking
contradiction. No new full model suite, performance matrix or BF16 qualification
was necessary for this documentation/artifact-only change.

For a clean installed reproduction, use a fresh checkout with this unchanged model
source, then run the following workload under the durable-job workflow if long:

```sh
source /usr/share/Modules/init/bash
module load ascend/dev-workspace-9.2.0
python -m venv --system-site-packages .venv-instances
. .venv-instances/bin/activate
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
python -m pip install --no-deps --no-build-isolation .
python -I examples/train.py --threads 4 --checkpoint NEW.pt
python scripts/check_upstream.py
python -m benchmarks.run --help
```

Use a new checkpoint path; `-I` makes the example consume the installed package.
The exact frozen launch/probe source and observations are also embedded in the
JSON record. Retired absolute paths there describe historical inputs, not live
dependencies. The next implementation programme starts only in the new thread.
