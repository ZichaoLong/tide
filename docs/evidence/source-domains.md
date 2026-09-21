# Logical source-domain qualification

2026-09-22 (Asia/Shanghai), clean implementation
`fb55f2e92e4979fc36e4bef1cf81172988b2df8f`.
**3480 tests passed in 265.18s**, including 184 new source-domain cases.
Both artifacts/source-domain-20260921-1958/status.json and
verification/result.json passed, exit 0, empty dirty status at that source.
Unit tide-foundation-source-domain-20260921-1958 is inactive/dead, MainPID 0,
Result=success, ExecMainStatus=0; finished 2026-09-21T20:02:49Z.

Command: `python scripts/qualify.py --output-dir
artifacts/source-domain-20260921-1958 --jobs 2`, through scripts/job.py in
background.slice. CPU aarch64, Torch/LibTorch 2.10.0+cpu, Python 3.11.15,
C++11 ABI, two build workers, OMP/OpenBLAS single-thread, backend autoload off.
The build manifest protects source/binary hashes. Graph identity is v12;
checkpoint payload remains v4. Different/earlier graph identities are rejected.
This gate reran all CPU tests and standalone C++ extension checks. Original LH
oracles were not rerun; their separate whole-model evidence remains at `776fc2e`.

## Contract exercised

See [source-domains.md](../source-domains.md). Incoming logical slots are dense
per node and may alias physical ports/edges. Physical CSR/CSC, bijective port
layouts, source tags, scales and ledgers remain distinct. Every complete fiber
has at most one atom per logical source; simultaneous aliases fail, even zeros.
All-source normalization and learned fiber-pooling vectors use the logical size.

An unsplit graph is compared with phase-exclusive aliases whose physical IDs
are deliberately permuted. Explicitly tied receive/send parameters are compared
across reference, Python frontier, native serial/parallel and packed/frontier.
Profiles: sum, mean, weighted mean, active/all softmax Aggregate, and
linear/active/all-softmax fiber pooling. Both dtypes include ragged batch 3,
width 4, two attention heads, absent sources and present zeros.

Check complete trace, contributions, state/cache order, selection/history,
emitted/pending messages and all cuts of a cyclic graph; HARD/HST/SOFTP and
selected clear vary in that cyclic gate. Independent public-root VJPs (including
zero cotangents), directly differentiated quadratic loss, optimizer updates,
shared-parameter checkpoint/resume, domain identity rejection, SettleGraph
boundary embedding and inference without semantic replay also pass. A closed
form test distinguishes logical all-source versus active-source denominators
and absent parameter gradients. Physical tag/receive-parameter retention is
checked separately; this is not a proof for arbitrary tag-sensitive embeddings.

## Development failures retained

artifacts/source-domain-python-20260922-a preserves an initial Python FP32
failure with its source archive/hash, command result and log. The test applied
another square to an already quadratic aggregate objective. Tiny upstream
rounding differences then crossed the declared gradient tolerance. Direct
quadratic-loss VJPs used less than 0.046 of that tolerance in the diagnostic.

artifacts/source-domain-dev-20260921-1951 preserves the successful native build
and 241 passes plus one analogous FP32 Settle test failure. The original dirty
tree is archived and this job remains failed. New acceptance isolates actual
output/state/pending tensors, including structural gradient absence, and
differentiates the quadratic loss directly. Runtime and tolerance were unchanged.
The corrected 250-test targeted run passed before the clean commit above.

This establishes the source-domain prerequisite, not local-clock behavior,
single-PDG LH containment, arbitrary pretrained import or sparse-scale speed.
