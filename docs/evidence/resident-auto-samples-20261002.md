# Automatic resident sample admission

Implementation source `3c3b4e7`; the [audited record](resident-auto-samples-20261002.json)
pins the full revision, installed consumer, unchanged backend dependencies,
immutable tests and complete trajectories. Audit:
`TASK/launchers/auto_samples_evidence.py SOURCE`.

Both resident consumers and the offline planner accept `--auto-sample-chunks`.
Before allocating the model they try the requested physical sample ceiling,
including existing operator-row reductions. Only a static memory refusal causes
sample halving with upward rounding; every attempt accounts for all logical
samples' saved continuations and gradient accumulation. The first fit is used;
one-sample refusal, invalid geometry and overflow terminate explicitly. Fixed
sample selection remains the default. The option changes neither the model nor
logical batch, precision, KV, queue, journal, window or update boundaries.

Clean-source **CPU 9 / NPU 18 checks passed**, without skips. CPU coverage includes
C++/Python planner agreement, uneven batches, the weakest card, storage and
accumulation charges, overflow, one-sample refusal, CLI and boolean validation.
NPU coverage includes eight automatic-sample cases across both clients,
FP32/FP16 and inference/training; eight prior fixed-sample/operator-splitting
cases; explicit pre-construction refusal; and the option contract. All sixteen
numerical cases compare complete continuation, gradients and updates against an
independent CPU schedule. Automatic training selects 9-row groups for logical
B17, including its partial tail, over two connected windows and two AdamW updates.

The first development run retained eight capacity failures because the new B17
test declared a queue of only 128. Raising that test's explicit capacities fixed
the fixture; runtime code and numerical tolerances were unchanged. That run
remains failed. This qualification rebuilds/verifies only affected consumer
objects; core libraries, public ABI and device kernels are unchanged.

This bounded admission heuristic does not guarantee optimal throughput or fit
for future data-dependent logical capacities. It does not use a numerical
prepass or OOM search. Full-size training and performance comparisons remain
separate; no new throughput or profiling claim follows from these tests.
