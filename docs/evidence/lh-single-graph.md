# Original LH inference encoded in one PositiveDelayGraph

Clean source: c84abbed7c9cfce6d4c6a44f6b7ae5f9aab465d3.
Full qualification passed: CPU regression **3946 tests in 326.58s**, original
Selector/Add/Full/Attention/Pronounce/IOCortex gates and independent Python
fixtures. All seven records under artifacts/single-qualified-20260921-2234/
are passed, exit 0, empty dirty state at that source. Unit
tide-foundation-single-qualified-20260921-2234 ended inactive/dead, MainPID 0,
Result=success. Started 2026-09-21T22:33:32Z; finished 2026-09-22T00:10:46Z.

```sh
python scripts/job.py --output-dir artifacts/single-qualified-20260921-2234 -- \
  python scripts/qualify.py --output-dir artifacts/single-qualified-20260921-2234 \
  --jobs 2 --lh-snapshot artifacts/lh-source-20260921-1428
```

The actual Python is /home/zlong/anaconda3/bin/python. Source ran from the frozen,
read-only /var/tmp/zlong-graph-execution-foundation/qualification-20260921-2234
with an independent build; artifact/snapshot paths were absolute local paths.
CPU aarch64, Torch/LibTorch 2.10.0+cpu, Python 3.11.15, C++11 ABI;
ATen/BLAS one thread, two build jobs, Nice=10/background.slice. Graph v13 /
checkpoint v5. The map adds an oracle-layer construction, no executor primitive.

## Qualified construction

For L body ticks per token, a period L+1 reserves the last phase for readout.
Body wires have phase-specific delay one/two; output wires have delay L-phase.
Periodic local state clocks exclude reserved readout ticks from body decay.
SourceDomain preserves the original logical coefficients across phase-exclusive
physical wires. Every phase projection aliases its original parameter tensor.
The rectangular vocabulary head stays outside uniform-width message payloads.

Every global cut compares projected body traces/states/history/ledger/pending,
partial-window buffers, readout state/history and labeled hidden/logits.
The original C++ think/think_single_step calls are unchanged; actual greedy
feedback is included in the token scenarios. Native construction is edge-major;
independent Python construction is phase-major, with different physical IDs.
The single graph runs native serial/parallel/packed and whole versus incremental
cuts. Existing CPU tests additionally exercise the owned cursor and sparse phases.

| Original assertions / dtype | Configurations | Single-PDG cuts | Known unavailable |
| --- | ---: | ---: | ---: |
| on / FP64 | 180 | 4950 | 24 |
| on / FP32 | 204 | 5610 | 0 |
| off / FP64 | 204 | 5610 | 0 |
| off / FP32 | 204 | 5610 | 0 |

Original FP64 active-softmax multi-batch diagnostics use an FP32 denominator.
The 24 assertions-on configurations remain explicitly unavailable; the
assertions-off build supplies their numerical comparisons. Both builds retain
ordinary C/C++ asserts. LH sources were not patched.

Independent Python: **48 fixtures / 28680 original candidate events / 660 cuts**.
Both native assertion variants export exactly 48 hashed fixtures. The terminal
audit in qualification-audit.json rechecked 513 hashes covering build binaries,
original snapshot files, fixture inventories, Python sources and the oracle result.
Snapshot identity (69 files):
ac7c878a56aeb55eec9f919da1962be6134fc5e3d1872f6d2303917306edb87f.
Original HEAD: 5fd237d40c9880ccb6e511e4bf20799c7022fd1e, plus recorded actual
user-modified source hashes. Reference repositories remain untouched.

## Precise boundary

This qualifies the equal-width, fixed-inference-weight homogeneous profiles
and scenarios of ../lh-iocortex.md. It establishes a bounded inference encoding,
not unrestricted containment of arbitrary LH configurations or model imports.
Readout projection deliberately omits the two-clock adapter's External.position
occurrence ledger. Missing phases make occurrence count different from token
number. The single graph's own continuation is complete; the projected readout
view is state/history/output correspondence, not full two-clock continuation.

Keep payload tolerances FP64 1e-10/1e-8 and FP32 1e-6/1e-5; routes are exact.
Each FP64 norm Read must match its own candidate; cross-norm differences are
bounded by actual candidate L2 error plus FP64 roundoff. No payload tolerance is
widened. Phase replication costs O(L*E). Unequal widths, arbitrary mixed profiles,
full imports and large-sparse performance remain unqualified.

LH is not a training authority. Separate Tide training/single-graph resume
qualification is in single-graph-training.md. Composite two-clock serialization
and standalone C++ optimizer ownership remain independent obligations.

During shared-storage exhaustion the correctness service was paused by
SIGSTOP/SIGCONT from 2026-09-21T23:14:08.216657Z to 23:14:11.782289Z while its
Git storage was relocated and verified. Source and build remained unchanged;
all terminal stages passed. This job is not timing/performance evidence.
