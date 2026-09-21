# Current handoff

Updated: 2026-09-22 (Asia/Shanghai). Branch: graph-execution-foundation.

## Verified baseline

3296 tests passed in 303.80s on clean implementation
`e6188802d3e893fa31d5351a9602748e0543915c`; evidence/pronounce.md.
Unit tide-foundation-pronounce-20260921-175646 is inactive/MainPID 0/exit 0.
All five records under artifacts/pronounce-20260921-175646/ passed cleanly:
outer status, verification, oracle, oracle-release and pronounce-release.
Actual Pronounce defaults to identity norm; its head remains outside the graph.

## Completed IOCortex development gate

Unit tide-foundation-iocortex-dev-20260921-1832 is inactive/MainPID 0/exit 0.
Outer status, oracle, oracle-release and python/result.json all passed.
Both original assertion variants in FP32: 204 configurations, 243780 candidates,
961188 messages, 3264 token/sample logits each. Independent Python: 24 exported
fixtures, 14340 original candidate events, body time-major and readout
reference/frontier, whole/cut continuation. This development run requested only
FP32. Source archive and logs are retained in its artifact directory.

Implementation: lh-iocortex.md. Actual original think/think_single_step, four
CSR/CSC blocks, explicit source/output slots, parallel physical edges, candidate
proposal/cache, selected activation, final physical state and selector counts,
all emitted/pending messages and logits. Native serial/parallel/packed and
original PACKED single/multi/CROSSBATCH; equal-width scope and two graph clocks.
No core runtime/schema changes and no whole-model training claim.

Retain failed iocortex-dev-20260921-1820 (two missing namespace aliases) and
failed iocortex-dev-20260921-1822 (180 FP64 cases passed, then the FP32 oracle
incorrectly applied FP64 tolerance to upstream FP32 norm perturbations).
The corrected test separately checks the FP64 norm formula, payload tolerance
and actual perturbation bound; discrete routes stay exact. New Python checker
also rejects -O and detects source/oracle/fixture changes; its -O probe exited 2
without creating an output directory.

## Clean qualification dispatch and next action

This implementation is being committed before the clean dispatch:
unit tide-foundation-iocortex-20260921-1855, output artifacts/iocortex-20260921-1855/.
Command: python scripts/job.py --output-dir artifacts/iocortex-20260921-1855 --
/home/zlong/anaconda3/bin/python scripts/qualify.py --output-dir
artifacts/iocortex-20260921-1855 --jobs 2 --lh-snapshot
artifacts/lh-source-20260921-1428.

Freeze source and shared caches while active. Inspect unit and all SEVEN terminal
records: status.json, verification/result.json, oracle/result.json,
oracle-release/result.json, pronounce-release/result.json,
iocortex-release/result.json and iocortex-python/result.json. No clean IOCortex
qualification result is asserted yet. Do not call a live job passed.
After success write evidence/lh-iocortex.md with the actual clean source, scope,
test/oracle counts and unavailable cases; update links/status in a separate commit.
Then continue lh-iocortex-plan.md: single-PDG clock embedding and logical source
domains, with complete-cut projections. Composite checkpoint/alias ownership and
large-sparse state/history/performance remain separate roadmap obligations.

## Runtime, reference and retention

LH snapshot artifacts/lh-source-20260921-1428: 69 files, identity
`ac7c878a56aeb55eec9f919da1962be6134fc5e3d1872f6d2303917306edb87f`; original HEAD
`5fd237d40c9880ccb6e511e4bf20799c7022fd1e`, including actual dirty-source hashes.
All reference trees are read-only. Keep this snapshot and all cited artifacts.
Original FP64 active-softmax diagnostic hardcodes an FP32 denominator; on builds
report the unavailable cases, off builds fill numerical coverage. Both retain
ordinary C/C++ asserts. Caches: build/lh-oracle and build/lh-oracle-release.

Prior retained failures/successes: pronounce-dev-20260921-{1744,1751,175316},
fiber-pool-dev-20260921-{1709,1715}, fiber-dev-20260921-{1601,1608}; their reports
retain causes and boundaries. Artifact cleanup dry run found no eligible entries;
no files removed this iteration. No push. STATUS is the only current handoff,
ROADMAP the backlog; source and evidence use separate commits.

CPU aarch64, /home/zlong/anaconda3/bin/python, Python 3.11.15,
Torch/LibTorch 2.10.0+cpu, C++11 ABI. Backend autoload off; OMP/OpenBLAS=1;
two build jobs, Nice=10, background.slice. Graph v11, checkpoint v4.
Packed training has tested first-order public-root VJPs with scalar replay;
inference does not replay. Higher-order AD and large-sparse speed remain unclaimed.
