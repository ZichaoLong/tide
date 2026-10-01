# Compact physical projection shard qualification

Implementation `acb84f30475c03e91cba5c35964565ab1b6316cf`; immutable clean source.
[Machine-readable audit](resident-projection-shards-20261002.json).

Explicit Full placement now also owns compact physical slot-affine projection
banks and their gradients. A device service packs actual row IDs and vectors,
sends work to the corresponding owner, skips empty owner subsets, and returns
vectors with device completion notification. It does not assemble a dense bank on
the coordinator. Shared immutable update snapshots, canonical alias reduction and
optimizer publication preserve this placement. No explicit placement retains the
existing dense path; callers can configure state placement independently.

Nine qualification jobs passed: both backend builds, installed standalone client,
31 library checks,48 standalone trajectories (768 windows/192 updates),27 actual
consumer checks,and independent two-device profiling. No skips, tolerance changes
or fallback warnings were accepted. Standalone trajectories include FP32/FP16,
three families,both schedules,retained windows and 1/2/3-device restoration.
Actual consumers cover12 FP32 inference and12 complete-training trajectories,
plus CLI/refusal/int64-boundary checks,through Python-owned and standalone runtimes.

Three new family cases compare full observables, gradients,three nonzero AdamW
updates and optimizer restore against independent CPU autograd. Separate FP32/FP16
cases run without diagnostic export with the entire projection bank on the remote
owner. Per-device bank reservations in the small mixed-placement cases are:

| Family | Logical device 0 | Logical device 1 |
| --- | ---: | ---: |
| PDG | 400 bytes | 400 bytes |
| TimedDAG | 400 bytes | 320 bytes |
| SettleGraph | 320 bytes | 320 bytes |

These demonstrate physical partitioning,not large-model memory measurements.
The separate D32 Attention/two-card training trace observes the projection packing
kernel on both devices:15,615 AI_VECTOR_CORE,631 AI_CORE,218 MIX_AIV tasks and no
observed AiCPU. It includes four continued windows,two AdamW updates and
construction. No formal throughput or speedup is inferred.

Builds reuse byte-verified unchanged core/kernels and dependency-verified affected
objects,then freshly link. Installed consumer object reuse also checks source,
public headers and compiler options. This is audited incremental qualification.
Two failed clean01 build attempts are retained: an external reuse launcher assumed
a copied kernel archive; clean02 follows the recorded artifact path and hash.
The retained library-dev02 failure was a CPU oracle fixture missing packed=False;
its correction did not change backend numerics or relax comparisons.

Total per-device memory admission,actual consumer FP16 and the full-size F6 matrix
remain pending. Current byte statistics are reservations,not allocator peaks.
