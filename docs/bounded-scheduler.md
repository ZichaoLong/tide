# Bounded device scheduling

This optional historical-workload consumer passed the finite
[84-cell CPU/NPU qualification](evidence/bounded-scheduler-qualification-20260929.md).
Full-size performance and capacity work remain in STATUS and ROADMAP D1-D6.
It does not replace the public eager executors, their independent CPU schedules
or checkpoint contracts.

## Finite contract

The program targets fixed Add/Attention topology, hard Full, observe-all/adopt/
clear, model FP32 Read and controls, positive physical-wire delays, and a finite
sequence beginning from empty state. Payload supports FP32/FP16. Static topology
and the finite clock window are expanded once. Presence, candidate and selection
masks, int64 selected/affected histories, cache visibility and Next decisions are
tensors. An absent message remains absent even when its padded payload is zero.
Physical edge identities and parallel edges remain separate.

Static predication can calculate inactive rows; logical Full/Emit occurs only
where device masks permit it. Add decay repeats multiplication per elapsed local
tick. Attention retains complete same-fiber queries, old/current KV visibility,
sequential bias decay and the all-source softmax denominator. Explicit workspace
limits reject unsupported capacities. The benchmark checks the topology-only
refusal bound before allocating model weights. The estimate is conservative and global;
it is neither a measured allocation nor a guarantee against per-device OOM.
`--workspace-gib` is a refusal threshold for this estimate (default16, maximum
32768), not an allocation or a device-memory reservation. Host RSS has its own
launcher bound; requested devices and their real allocators bound physical HBM.
Large thresholds are useful only for explicitly bounded capacity experiments.

Dynamic topology, arbitrary imported state, checkpoint/resume, unbounded queues,
HST, higher-order AD and arbitrary custom programs are outside this interface.
No backend silently substitutes an algorithm or copies scheduling decisions to
CPU. Tensor/scalar export is an explicit window boundary.

## Differentiation and updates

Ordinary autograd of predicated arithmetic includes padding. The explicit
first-order VJP carries a boolean structural dependency set per sample/leaf.
Branches select semantic dependencies; connected-zero clear and empty-cache
slices retain theirs. Gradient export uses that set to restore None versus zero
and rejects a nonzero derivative on a disconnected leaf. Exported ordinary Result
tensors are detached numeric observations; differentiation uses Value/VJP.

SGD/AdamW use FP32 masters and slots. Device masks skip disconnected parameters,
weight decay and owner counters, while connected zeros update normally. A device
finite preflight gates the complete update. AdamW correction coefficients are
prepared using host double arithmetic and selected by exact device counters from
a finite table, matching the eager reference's scalar rounding policy. Exceeding
the declared update capacity fails through the device status flag. FP16 uses a
static loss scale. Tolerances are explicit; FP32 defaults and exact discrete
checks cannot be relaxed by the FP16-only check options.

## Replay and cross-device execution

Single-device execution captures a fixed window in an optional native NPUGraph
adapter. Python is not a runtime dependency. Replay launches the captured device
program; there are no per-event host scalar or index decisions. The host still
owns input/output boundaries, graph construction, launches and status reporting.
Integer sorting can execute on device AiCPU on the current stack; device
residence does not imply that every operation uses AiCore.

The experimental multi-device adapter creates one model per device. Each peer
transfer uses a cross-device IPC Notify after the immutable source clone is ready;
the destination waits, resets its notification and pulls the peer data on the
consumer stream. Copy and consumption then share the destination ordering domain. Warmup creates a finite
notification inventory (currently at most16384 channels). Capture must reuse that
inventory and shape/dtype/pair signature. Raw-API buffers remain owned for the
captured window. This has additional memory cost. Missing runtime APIs or peer
capabilities fail explicitly; ordinary Torch cross-device copy is not substituted
inside captured execution.

Ordinary cross-device autograd inserts Events which conflict across captured
models on the tested stack. The experimental adapter therefore cuts only the
physical communication edges, records source/destination pairs and explicitly
propagates cotangents in reverse topological order with the same Notify protocol.
Torch differentiates device-local subgraphs. The final parameter VJP combines
all local roots before the structural dependency mask is applied. This audit
implementation can traverse local graphs repeatedly; its cost must be measured.
A passing communication primitive is not complete scheduler/training acceptance.

## Entry points and evidence boundary

`tide-bounded-schedule-check` compares full observations and isolated VJPs with
an independent scalar CPU schedule. Isolated VJPs use two deterministic external
cotangents, independent of the observed output, plus a connected-zero probe.
Both sides receive the same exact binary fractions. This keeps the Jacobian
comparison separate from rounding in upstream-loss expressions and from the
cancellation-sensitive radial direction of a normalized output. Complete training
losses and their gradients are checked separately.
It also checks three complete SGD/AdamW
updates, master/slot values and owner counters. One captured update is replayed three
times with persistent owners; trials reset owners and change inputs. Optimizer
guards separately check absent owners, connected zero, missing/nonfinite
gradients and the declared update limit. Isolated multi-device roots use the
same explicit communication VJP tape outside capture.
`--peer-check 1 --devices 2` isolates strided/scalar/int64/bool communication and an
analytic VJP with unused and connected-zero owners. Float payloads, small bool
masks and int64 values above2^55 change across replay trials. The ordinary narrow FP16
check remains the default; any relaxed envelope is recorded in the command.

`tide-bounded-scale` and `scripts/benchmark_bounded_scheduler.py` measure complete
reset-state windows in eager or replay mode, with persistent training owners.
The wrapper requires a clean source/build identity, finite wall/RSS bounds,
portable records and declared tracking. Construction, warmup/capture preparation,
CPU input-ID construction, finite checks and metrics are outside window timing;
input upload and complete forward/optional VJP/optimizer are inside. This scope
must not be equated to an existing token-warmup latency table. FP32/FP16 pairs
must use the same topology, dimensions, placement, window and timing scope.
Metrics also report locality cut edges, the custom peer notification inventory
and retained peer-buffer bytes. Those bytes exclude other graph workspaces and
are not a complete transport counter.

Full-size performance requires its own capacity and correctness assessment.
Small-tensor qualification does not certify fit, speed or numerical routes for
D2048/B512/V50304 or for another CANN/device stack.
