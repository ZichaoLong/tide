# Bounded single-PDG training and value resume

Clean source: d233429cd5807614869214dcea21d9492e309fd1.
**4901 CPU FP64/FP32 tests passed in 584.15s**, exit 0, empty dirty state.
CPU aarch64, Python 3.11.15, Torch/LibTorch 2.10.0+cpu, C++11 ABI;
ATen/BLAS one thread, two build jobs, Nice=10/background.slice.
Graph v13/checkpoint v5. The optimizer/checkpoint controller is Python;
execution and autograd include native serial, node-parallel, packed and cursor.

Unit tide-foundation-training-qualified-20260921-2345 ended inactive/dead,
MainPID 0, Result=success, exit 0. Source was frozen/read-only in
/var/tmp/zlong-graph-execution-foundation/training-qualification-20260921-2345
with its own build. Started 2026-09-21T23:54:46Z, finished 2026-09-22T00:09:41Z.
The verification stage ran 23:59:54Z–00:09:40Z. Every recorded build-binary hash
and the frozen source/clean state were checked after termination.

```sh
python scripts/job.py --output-dir artifacts/training-qualified-20260921-2345 -- \
  python scripts/qualify.py --output-dir artifacts/training-qualified-20260921-2345 --jobs 2
```

Actual interpreter: /home/zlong/anaconda3/bin/python. Output resolves to
/var/tmp/zlong-graph-execution-foundation/artifacts/training-qualified-20260921-2345/.
Inspect status.json, task.log, verification/result.json and verification/tests.log.
Native module SHA256:
e86bbd4f9483b325a264a8b3c065f4b02a6abd2d0a3db062cc5b4d811738f0c7.
This CPU gate does not itself run original LH; that oracle has separate evidence.

## New coverage

The 940 added cases qualify the bounded map under **Tide training semantics**:

- 360 isolated-root comparisons: six Add/fiber profiles, HARD/SOFTP/HST, clear
  on/off, both dtypes, Python and four native schedules. Check full forward
  body projection, readout state/history/output and unfinished windows; compare
  independent input/parameter VJPs for logits, state/slots and pending payloads.
- 60 zero-cotangent cases preserve None versus connected zero; 20 cut cases
  preserve input VJPs without implicit truncation.
- 240 three-update SGD/AdamW comparisons preserve shared phase/original-edge
  owners, gradients, weights and complete optimizer state. The rectangular
  vocabulary head is unused before the first readout and active afterwards.
- 20 truncation cases demonstrate that detaching both graph continuations while
  leaving the partial-window buffer connected gives matching forward values but
  incorrect old-input gradients.
- 240 resume cases save and restore twice inside nonempty token windows, using
  the complete encoded graph checkpoint. Fresh reconstructed weights are altered
  before restore, then aliases, detached caches/messages, optimizer state and
  subsequent gradients/updates must match the uncheckpointed two-clock anchor.

Each phase replica must use the same Tensor object as its original owner;
the encoded and two-model parameter identity sets must be equal. Existing
executor, profile, checkpoint, gradient-connectivity and benchmark tests reran.
Independent schedule and exact projection details are in ../single-graph-training.md.

## Limits and retained failures

AdamW is explicitly lr .0002, epsilon 1e-5, weight decay .01; SGD is lr .001,
momentum .8, weight decay .01. HST uses zeta .37. Tensor tolerances remain
FP64 1e-10/1e-8 and FP32 1e-6/1e-5; identities/routes are exact.
Default-epsilon FP32 packed AdamW amplification is retained in
artifacts/single-training-adamw-fp32-repro/ and is not certified by this matrix.
No comparator tolerance was widened. The initial invalid buffer-presence test
assumption is retained in artifacts/single-training-initial-buffer-repro/.

These are short synthetic recurrence/update checks, not model-quality evidence.
No claim covers LH training, unequal widths, arbitrary model import, higher-order
AD, a composite two-clock/controller checkpoint or a standalone C++ optimizer
format. Readout projection omits the adapter-only occurrence ledger; the single
PDG's own continuation is complete. Later atomic publication source 00bbf78 has
its own bounded qualification in checkpoint-io.md; it is not part of this source.
