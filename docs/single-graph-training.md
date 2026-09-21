# Training correspondence for the bounded single-PDG map

This is Tide two-clock versus Tide single-PDG training. LH supplies the original
inference topology and local formulas; its training is not an authority. The
inference construction and readout-ledger projection remain in lh-single-graph.md.
No runtime/schema change is introduced by this validation increment. Current
qualification status belongs in STATUS; exact evidence follows the clean gate.

## Owners and differentiable boundary

The application owns the body and readout model together. Distinct physical
phase edges alias the original send/receive and local projection tensors.
Parallel original edges may themselves share parameters. The encoded model must
have exactly the same set of Tensor identities as the two application models;
extra, dropped or copied parameters fail. A registered rectangular 7-by-4
vocabulary head remains outside the uniform-width graph.

A complete two-clock training cut owns body continuation, readout continuation
and every body output in the unfinished token window. The encoded graph stores
that last category as pending messages. Losses on pending payloads must include
it on both sides. No synthetic readout occurrence ledger is inferred from time.
Readout state/history and labeled outputs are compared under the declared
projection; complete two-clock serialization is a separate obligation.

In-memory cuts preserve autograd. At a truncated update boundary, detach both
continuations AND the partial-window buffer (or the single graph's whole
continuation/cursor). Then update shared parameters once using stable deduplicated
application owners. A negative control deliberately keeps the buffer connected:
forward values still agree, but old input gradients survive. This is why numerical
forward checks alone cannot establish a training boundary.

The tick-repeat state profiles keep their specified encoded state under parameter
updates. This validates Tide's recurrence across weight epochs; it does not assert
that old LH eager decay under changing weights is the same computation.

## Independent schedules and tests

- tests/single_graph_training.py owns a literal two-clock Python schedule and
  the single-PDG adapter, plus explicit alias checks. The encoded scheduler is
  independently Python streaming or native serial/parallel/packed/cursor.
- tests/single_graph_roots.py roots individual logits, combined logits, body
  and readout state/slots, one in-flight body message and one partial-window
  payload. Independent external leaves expose cross-sample gradient leakage.
- tests/test_single_graph_training.py checks six Add/fiber profiles, HARD,
  SOFTP and HST (zeta 0.37), selected clear on/off, FP64/FP32, values, routes,
  gradients and unchanged connectivity through cuts. Separate zero-cotangent
  cases retain None versus connected-zero observations.
- tests/test_single_graph_optimizer.py checks three truncated updates with
  SGD (lr .001, momentum .8, weight decay .01) and AdamW (lr .0002, epsilon
  1e-5, weight decay .01), Add/all-softmax, all three Emit modes, both clear
  settings and all five encoded schedules. Compare gradients, parameters and
  complete optimizer states at every update. The first cut precedes readout,
  so head parameters must be skipped; later cuts must connect them.

The optimizer controller is Python over aliased LibTorch tensors for native
execution. This does not qualify independent C++ optimizer serialization or a
composite application checkpoint. It does not cover arbitrary topology import,
higher-order AD, all pretrained architectures or long training trajectories.

## Numerical qualification boundary

Tensor tolerances remain FP64 1e-10/1e-8 and FP32 1e-6/1e-5; discrete routes
remain exact. The AdamW epsilon above is an explicit workload setting, not a
comparison-tolerance change or a backend fallback.

A retained default-epsilon (1e-8) FP32 failure is in
artifacts/single-training-adamw-fp32-repro/. Packed cursor query gradients around
6e-10 differed by about 7e-11; default AdamW amplified this to a 1.24e-6 parameter
difference, exceeding the strict parameter check. Gradient maximum difference
was 7.45e-9; the corresponding FP64 native parameter differences were below
3e-15. Python single-PDG and two-clock calculations agreed exactly in the
reproducer. The archived source, failing log and serial/packed/dtype diagnostic
remain retained. Default-epsilon FP32 multi-update parity is not qualified by
the explicit-epsilon matrix; no tolerance was widened to label it passed.
