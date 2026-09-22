# Two-clock token checkpoint and strict Python coordinates

Clean source: 69ca37900e9c10d3fca95570ea1ebca8f9079f46.
**6233 CPU FP64/FP32 tests passed in 665.45s**, exit 0, empty dirty state.
CPU aarch64 Linux, Python 3.11.15, Torch/LibTorch 2.10.0+cpu, C++11 ABI;
ATen/inter-op/OpenBLAS one thread, two build jobs, Nice=10/background.slice.
Graph v13/single-graph checkpoint v5 remain unchanged. The separate application
format is tide-token-application-v1. Optimizer and file ownership are Python;
execution/autograd include independent Python and native serial/parallel/packed.

## Reproduction and integrity

Unit tide-foundation-token-coordinates-qualified-20260922-0106 ended
inactive/dead, MainPID 0, Result=success, exit 0. Started 2026-09-22T01:05:00Z,
finished 01:21:19Z. Verification ran 01:10:11Z–01:21:19Z. Frozen source/workdir:
/var/tmp/zlong-graph-execution-foundation/qualification/token-coordinates-20260922-0106.
It used an independent build. All 339 tracked files matched git archive after
exit and remained read-only; all eight recorded build binary hashes matched.

```sh
python scripts/job.py --output-dir ABS_OUTPUT -- \
  python scripts/qualify.py --output-dir ABS_OUTPUT --jobs 2
```

Resolved Python: /home/zlong/anaconda3/bin/python. ABS_OUTPUT:
/home/zlong/llm/graph-execution-foundation/artifacts/token-coordinates-qualified-20260922-0106.
Retained status.json, task.log, verification/{result.json,tests.log}, dispatch.json
and qualification-audit.json bind commands, clean source, environment, inventory,
source/binary hashes, terminal state and counts. Native module SHA256:
e86bbd4f9483b325a264a8b3c065f4b02a6abd2d0a3db062cc5b4d811738f0c7.

## Application checkpoint coverage

248 new cases exercise the complete two-clock boundary:

- 192 comparisons perform three truncated SGD or AdamW updates, saving/restoring
  twice inside nonempty partial token windows. Compare uninterrupted Python with
  restored Python/native serial/node-parallel/packed execution, Add/all-softmax,
  clear on/off, HARD/SOFTP/HST and FP64/FP32. Both complete continuations, actual
  occurrence ledgers, window buffers, traces, gradients, weights and optimizer
  state are checked. Send/receive and memory parameters share cross-graph owners.
- 34 corrupted-file cases reject identity/policy, missing ledger, component
  clocks/batch, second-graph pending messages, buffer coordinates/duplicates/
  tensor metadata, shared values and optimizer order/shape before mutation.
- 22 additional cases cover occurrence positions differing from token time,
  detached buffers/caches, lost aliases, missing optimizer state, policy mismatch,
  rejected save and explicit boundary configuration.

The application stores both input occurrence ledgers directly; it does not infer
one from time or claim reconstruction from the single-PDG projection. Single-
graph v5 codec extraction preserves its public API/serialized keys. AdamW retains
the explicit epsilon 1e-5 from the prior training contract. No tolerance changed:
FP64 atol/rtol 1e-10/1e-8, FP32 1e-6/1e-5; identities/routes and None/zero VJPs
remain observable. Contract: ../token-application-checkpoint.md.

Prior development gate: artifacts/token-bundle-dev-20260922-0038/, 1398 passed /
342.38s. Its dirty-source archive contains 335 files, compared byte-for-byte to
the unchanged tree before commit; post-run-audit.json retains the archive hash.
That development result is distinct from this clean full qualification.

## Coordinate correctness

The previous Python reference accepted time=0.5, updated the input ledger and
silently produced no event; native conversion rejected it. Both accepted bool
coordinates. artifacts/coordinate-types-probe/probe.json retains those results.

1074 new cases qualify strict exact-Python-int/int64 boundary checks:
160 malformed external records, 128 malformed window bounds, 704 imported
state/history/ledger/pending cases, 18 graph index/budget cases and 64 single/
application checkpoint cases. Executors include reference/frontier/chain,
native serial/parallel-packed/frontier/chain and owned cursor. Rejection must
precede mutation; corrected input or state can then execute and match reference.
Checkpoint rejection leaves all live weights/optimizer values unchanged and
invalid saves publish nothing. Existing counter-overflow/maximum-value cases
also pass. Cursor advances check incoming metadata, without scanning held state.

The test-fixture failure where bool dictionary insertion collided with an
existing int key remains in artifacts/coordinate-test-key-collision-repro/.
The corrected fixture removes the colliding key before inserting malformed data;
this repairs the test's corruption, not a numerical tolerance or runtime result.

This gate reruns all prior CPU numerical checks but does not rerun the original
LH full oracle (no C++/oracle change); its independent source/evidence remains
lh-single-graph.md. No broader model-import, higher-order AD, performance,
standalone C++ persistence, arbitrary controller/RNG/data cursor or power-loss
claim follows. Later durable-record tooling at 3604ec0 has separate evidence in
durable-records.md and was not part of this frozen graph qualification.
