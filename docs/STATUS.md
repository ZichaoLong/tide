# Current handoff

Updated: 2026-09-22 (Asia/Shanghai). Branch: graph-execution-foundation.

## Verified baseline

Clean implementation `776fc2e597dac872f4204ac3a09966db39ab74c0` passed
3296 tests in 240.23s and all original Selector/Add/Full/Attention/Pronounce/
IOCortex gates; see evidence/lh-iocortex.md. Unit
`tide-foundation-iocortex-20260921-1855` is inactive/dead, MainPID 0, exit 0.
All seven records in artifacts/iocortex-20260921-1855/ are terminal passed,
exit 0, clean at that source: status, verification, oracle, oracle-release,
pronounce-release, iocortex-release and iocortex-python. Finished
2026-09-21T19:28:09Z. No running jobs; source/shared caches may be edited.

IOCortex original assertions-on FP64: 180 cases and 24 unavailable (known LH
active-softmax diagnostic); assertions-on FP32 and assertions-off FP64/FP32:
204 cases each. Independent Python: 48 fixtures / 28680 original candidate
events. Actual original think/think_single_step, greedy feedback, all four
adjacency blocks, explicit slots, candidate/cache/physical hidden, both counters,
emitted/pending messages and logits match within the bounded two-clock scope.
Native serial/parallel/packed and original PACKED/CROSSBATCH are covered.
Each case uses one homogeneous body/readout profile. No core runtime/schema
change in this gate and no whole-model training or single-PDG claim.

## Source-domain increment and clean qualification

SourceDomain is implemented: graph v12 / checkpoint v4, logical incoming domains
separate from physical PortLayout, complete-fiber collision rejection,
Python/native factories and SettleGraph remapping. Contract: source-domains.md.
250 targeted checks passed in 35.73s (FP64/FP32), including standalone C++ custom
kernels, domain guards, reference/native schedules, cyclic every-cut cursor,
SettleGraph, isolated public-root/direct-loss VJPs, optimizer and alias save/resume.
Implementation is being committed before the clean qualification below.

Dispatch: unit tide-foundation-source-domain-20260921-1958, output
artifacts/source-domain-20260921-1958/. Command: /home/zlong/anaconda3/bin/python
scripts/job.py --output-dir artifacts/source-domain-20260921-1958 --
/home/zlong/anaconda3/bin/python scripts/qualify.py --output-dir
artifacts/source-domain-20260921-1958 --jobs 2. This gate reruns all CPU tests;
it does not rerun original LH oracles (their last qualified source is above).
Freeze checkout/shared build while active. Inspect unit, status.json,
verification/result.json and tests.log to establish terminal success. Then save
source-domain evidence and update STATUS/ROADMAP in a separate commit.

Retain failed artifacts/source-domain-dev-20260921-1951: native build passed,
241 tests passed and one FP32 Settle test failed. Its original dirty source,
hash and logs remain archived. Like source-domain-python-20260922-a, it squared
an already quadratic objective again. New tests independently root actual public
tensors and directly differentiate the quadratic loss; runtime and tolerance are
unchanged. Keep both failure records failed. The corrected 250-test run above
included all previous targets plus standalone C++ CLI/kernel checks.

Next: periodic local state clocks (body phases 0..L-1, token readout phase L,
period L+1), delegated step/batch/sequence contracts and global stored timestamps.
Prove cut conversion and idle decay without changing Full's global phase clock.
Then continue lh-iocortex-plan.md's single-PDG arbitrary-cut projection. Phase
occurrence ledgers cannot be inferred from token time: either explicitly track
them or state a narrower projection directly to original LH, which has no such
ledger. Composite checkpoint ownership and large-sparse performance stay pending.

## Runtime, reference and retention

LH snapshot artifacts/lh-source-20260921-1428: 69 files, identity
`ac7c878a56aeb55eec9f919da1962be6134fc5e3d1872f6d2303917306edb87f`; original HEAD
`5fd237d40c9880ccb6e511e4bf20799c7022fd1e`, including actual dirty-source hashes.
All reference trees are read-only. Keep this snapshot and all cited artifacts.
Original FP64 active-softmax diagnostic hardcodes an FP32 denominator; on builds
report unavailable cases, off builds fill numerical coverage. Both retain
ordinary C/C++ asserts. Caches: build/lh-oracle and build/lh-oracle-release.

Retain failed iocortex-dev-20260921-1820 (missing type aliases) and
failed iocortex-dev-20260921-1822 (FP32 oracle demanded FP64 upstream norm accuracy).
The corrected oracle preserves payload tolerance and exact routes; see evidence.
Successful iocortex-dev-20260921-1832 tested both original FP32 variants and Python.
Earlier retained records: pronounce-dev-20260921-{1744,1751,175316},
fiber-pool-dev-20260921-{1709,1715}, fiber-dev-20260921-{1601,1608}.
Their reports retain causes and boundaries. Last artifact cleanup dry run had no
eligible entries; no files removed. No push, no sub-agents. STATUS is the only
current handoff; ROADMAP is the backlog.

CPU aarch64, /home/zlong/anaconda3/bin/python, Python 3.11.15,
Torch/LibTorch 2.10.0+cpu, C++11 ABI. Backend autoload off; OMP/OpenBLAS=1;
two build jobs, Nice=10, background.slice. Working graph v12, checkpoint v4;
qualified IOCortex baseline used graph v11.
Packed training has tested first-order public-root VJPs with scalar replay;
inference does not replay. Higher-order AD and large-sparse speed remain unclaimed.
