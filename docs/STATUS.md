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

## Source-domain qualification completed; next local clocks

Clean implementation fb55f2e92e4979fc36e4bef1cf81172988b2df8f passed 3480 tests in
265.18s; evidence/source-domains.md. Unit tide-foundation-source-domain-20260921-1958
is inactive/dead, MainPID 0, Result=success, exit 0. Outer status and verification
records in artifacts/source-domain-20260921-1958/ are passed/clean at that source,
finished 2026-09-21T20:02:49Z. No live jobs; checkout/shared build may be edited.
This CPU gate did not rerun original LH oracles; their exact source stays above.

Graph v12 / checkpoint v4: SourceDomain separates logical incoming coefficients
from bijective physical PortLayout. Complete-fiber collisions fail. Python/native
factories, packed fiber caches, every-cut cyclic continuation, isolated VJPs,
optimizer/alias checkpoints and SettleGraph remapping are covered. Domain maps
leave physical tags and receive/send parameter ownership explicit. Source files
are small and navigable via docs/source-domains.md.

Retain source-domain-dev-20260921-1951 (build passed, 241 tests passed, one FP32
Settle test failed) and source-domain-python-20260922-a. Both failed tests squared
an already quadratic objective again. Current acceptance isolates actual public
tensors and directly differentiates the loss, without changing runtime/tolerance.
The final targeted run passed 250 tests including standalone C++ custom kernels.

Next: StateClock(period, first, count), a contiguous valid-phase interval per
period. Body = (L+1,0,L); readout = (L+1,L,1). This compact policy avoids a heap
allocation for every native node. Convert state-program event/last timestamps
into local ticks; return stored metadata to global time. Delegate step, batch,
sequence and packed capabilities; preserve tensors, observations and content
source tags. Keep Full phases, region timestamps and message delays global.
Guard invalid phases and inverse int64 overflow; validate checkpoint identity.

Then complete lh-iocortex-plan.md's single-PDG arbitrary-cut projection. Phase
occurrence ledgers cannot be inferred from token time: track them explicitly or
state a narrower projection directly to original LH, which has no such ledger.
Do not claim complete dual-graph continuation equality without these counters.
Composite checkpoint ownership and large-sparse performance stay pending.
No implementation work on local clocks has been applied yet.

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
