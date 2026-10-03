# Eager consumer memory admission

The Python/native and independent LibTorch Add/Attention consumers estimate
incremental memory before constructing their model. This is a consumer policy,
separate from core graph semantics, ownership and the online scheduler. Source
is in `tools/online_bench/eager_capacity.{py,h}` and
`traffic_bounds.{py,h}`. Implementation, qualification and calibration are
reported separately in [STATUS](STATUS.md).

The shared CLI accepts `--device-memory-bytes`, `--head-workspace-bytes`,
`--chunk-policy conservative|aggressive` and `--auto-sample-chunks` for CPU and
mixed eager execution as well as resident execution. The resident implementation
keeps its existing capacity model and operator splitting. `--resident-*` limits
and continuation pools remain specific to resident execution.

Offline planning loads no Torch or device runtime. It uses the caller's explicit
budget; run-time admission also checks live free memory. For example:

```bash
python scripts/plan_execution_flow.py --packet artifacts/flow-input/workload.json \
  --preset mixed-c --devices 2 --device-memory-bytes 64424509440 \
  --training --steps 2 --warmup 0 --windows-per-step 2 \
  --auto-sample-chunks --chunk-policy aggressive
```

The offline default preset remains resident for compatibility. A planned result
is an estimate and does not establish hardware availability, calibration or a
completed workload.

## Static inputs and limits

Admission uses a validated continuous consumer packet, an owner map, dtype,
optimizer, logical batch, independent physical sample rows, worker count and the
entire declared run horizon `(warmup + steps) × windows × tokens`. No model is
executed to derive events or routes. Learned parameter elements match the
consumer's independently initialized named tensors.

Finite DAG packet traffic is bounded per input position. Integer arrival-offset
intervals enclose all paths, including unequal delays. At each region/time at
most its declared selection budget can emit; destination and owner counts use
the largest outgoing multiplicities under that budget. Parallel physical edges
remain separate contributions. This avoids enumerating exponentially many paths.
The next input position begins after this packet's maximum readout offset.

These are upper bounds over legal selections, not measured activity or a
scheduler input. They apply only to the finite DAG consumer packet format with
empty initial state. They do not narrow the public PDG scheduler's support for
positive-delay feedback or change legal online batch formation.

Per-owner accounting includes learned parameters, shared fixed constants,
logical-batch persistent state/KV through the complete run horizon, gradients,
SGD momentum or AdamW slots, connected-window vector work and attention cache
proposals/scores, worker workspaces, head/loss, bounded transport and a backend
allowance. Construction also charges the named integer initializer's transient
arrays. A phase maximum is recorded for construction, forward, backward and
optimizer. Operator workspace declarations are limits, not allocations.
FP16 training additionally charges FP32 master parameters at construction and
throughout the run, plus FP32 master gradients/optimizer slots alongside payload
gradients. Its shape estimate is implemented; device calibration is pending.

CPU RSS additionally charges6.25% of learned and explicit master storage for host allocations and
retained buffers at each phase. The original-width B4 fresh-process
[calibration](evidence/original-width-eager-cpu-calibration-20261003.md) found up
to4.3% unmodelled learned-storage bytes at construction, and an Attention total
peak0.51% above the previous maximum. This explicit CPU allowance covers both
observations with margin; a new scale run must still validate it. Accelerator
allocated-byte estimates do not inherit CPU RSS overhead. Offline `--preset cpu`
and actual CPU resolution (including `--device auto`) select the RSS counter.

The shape coefficients are an explicit estimate requiring peak calibration;
they are not a proof about vendor/driver allocations. Current policy reserves
10% plus 128 MiB of each card budget for aggressive splitting, or 25% plus
128 MiB for conservative splitting. The head budget has the matching percentage
reserve. Both strategies use the same shape estimate.

## Splitting and failure

A positive device budget caps the initially observed free memory. Zero selects
current accelerator driver free memory, or at most half currently available CPU
RAM. CPU availability comes from `/proc/meminfo` with a supported system query
fallback. An unavailable explicit memory query fails rather than inventing a
capacity. Physical owners are resolved before model allocation.

Without automatic splitting, an over-budget plan is refused. With automatic
splitting, independently executable sample rows are halved, bounded below by
one. Every attempted shape is retained. Learned parameters, persistent logical
state/KV, dtype, logical batch, connected windows, loss denominator and optimizer
update boundaries remain unchanged. A refusal at one row is preserved as a
failure and does not establish full-size delivery.

The head envelope bounds all outputs in the connected physical slice; it
currently reduces sample rows rather than tiling the differentiable eager head.
The shape estimate does not reposition owners to hide an explicit map's excess.
Locality/learned-parameter placement is a separate deterministic static choice.

After execution, phase-boundary observations are compared with the estimate.
Accelerators report framework allocator peaks; CPU reports growth of the
process-lifetime peak RSS as a proxy. CPU peak reuse within one process can hide
new allocations, so scale calibration uses fresh processes. Driver allocations
outside the framework and changes by other users are not certified by this
counter. Measurement does not add per-event sampling to the timed loop.

Preallocation refusal writes the attempted plan and failure phase. A post-run
underestimate retains the full measurements and outputs while returning failure.
Neither path removes memory guards, silently truncates KV nor changes precision.

Directed checks enumerate all legal top-k choices on small nonaligned/parallel
graphs, compare Python/C++ static plans, force sample splitting and compare full
continued records/None-aware gradients/two updates against independent CPU
execution, and check preallocation and post-run failure records. Accelerator
calibration and full-size completion remain separate evidence.

Clean CPU70/NPU12+40 gates, original-width CPU Add/Attention RSS recalibration,
seven two-device calibrations and a separate actual profile are
[qualified on c686096](evidence/eager-consumer-capacity-20261003.md).
B512 CPU/mixed capacity and formal performance require their own execution.
