# Append-only resident fiber KV staging

Tested source: `80dae6e14d41614d0cdb1056bb39b57ca10d07ed`, clean immutable
snapshot. All seven build/gate/diagnosis jobs passed. CPU4, public NPU33,
four standalone component cells, no skips.
[Audit, exact source/binary identities and allocation observations](resident-fiber-append-20261002.json).

## Change and semantic boundary

Fiber attention previously allocated and copied two complete key/value proposal
banks even though each proposal only appends after the owner's published cache
length. It now stages these append rows in the unused live-cache tail. The
existing device plan checks capacity and assigns disjoint positions; adoption
publishes lengths only after the stage's checks. Rejected selection leaves the
old prefix and length intact. A later proposal overwrites the invisible tail.

Bias decay keeps separate proposal storage because it changes visible old rows.
The independent zero sentinel remains untouched. Selected clear publishes an
empty cache. Retained windows still clone their actual values and keep their
independent VJP/continuation contract. Device errors still poison the owner;
this change does not add rollback/retry after device execution failure.

No logical batch, KV/queue capacity, dtype, visibility, loss denominator or update
boundary changes. No host event loop or topology-specific shortcut is added.
Both complete-consumer capacity planners and the backend reserve subtract only
the exact removed payload banks, retaining all other conservative allowances.
Public ABI and CANN kernels are unchanged.

## Affected qualification

- CPU4 includes24 C++/Python shape plans, per-card admission/physical splitting,
  retention/precision costs and overflow refusal.
- Public NPU33 compares both standalone LibTorch and Python-owned resident
  clients against independent CPU execution: three families, FP32/FP16,
  streaming/prefill, continuous inference, complete multi-window SGD/AdamW,
  single/two-device execution, eight forced-splitting updates and early refusal.
  Full observable/gradient/parameter-update records accompany the training cases.
- Fiber forward:16 analytic anchors,192 windows,6 refusals and60 lifecycle
  windows. Added lifecycle cases alternate accepted/rejected proposals with
  changing payloads, including selected clear, periodic clocks and old KV.
- Node-time batching:96 anchors/restores,480 general-topology windows/restores
  and3 saturation cases, covering complete-fiber visibility and key tiling.
- Retained reverse:152 trajectories/608 windows **per dtype** in FP32 and FP16,
  independent FP32/FP64 CPU references, after-close/poisoned-source replay,
  None/zero roots, mixed caches, feedback, parallel edges and cache carry.

Development passed the same affected gates. Qualification reuses two compiled
host objects only after source/header identity checks, freshly links each
runtime, checks loader closure and byte-verifies unchanged core/kernel inputs.
The installed consumer reuses verified unchanged objects and links the new
backend. This is not a new full-core or CANN compilation. The unaffected8,954
CPU checks were not rerun. No numerical tolerance was relaxed.

## Separate allocation diagnosis

The unchanged representative Attention packet uses128 body nodes/544 edges,
D128/B8/T4/V257,17,384,240 parameters, FP32, one NPU, two continued windows,
one complete AdamW step and no warmup. Capacities match the representative
screen: queue/arrivals4096, outputs128, trace8192, KV128, KVtrace65536, aggressive.
Python owns the client of the C++/CANN backend in both runs. Instrumented
durations are excluded from throughput evidence.

| Observation | Before, clean2222d9d | After, clean80dae6e |
| --- | ---: | ---: |
| Peak allocated bytes | 2,560,387,072 | 2,426,168,320 |
| Allocated after construction, MiB | 800.746 | 672.744 |
| Allocated after two forward windows, MiB | 1366.032 | 1238.030 |
| Allocated after completed step, MiB | 801.583 | 673.581 |

Peak reduction is134,218,752 bytes, or128.001 MiB (about5.24%). It matches
the exact removed two payload tensors. Loss, outputs, final cut, event and
training statistics match exactly. Both observed peaks remain inside admission
estimates; every recorded allocation is eventually freed.

TorchNPU's bounded allocation history supplies useful C++ frame attribution.
Before the change, the peak's largest requested-byte categories include retained
tapes, cache adjoint seeds, forward KV proposals and optimizer storage. History
entry sizes describe requested bytes; allocator peak counters include rounding.
Both exclude vendor/driver allocations outside the TorchNPU allocator. Trace
length stays below100,000 and eight snapshots below the256MiB raw limit. This
is a capacity diagnosis, not a CANN operator-engine profile or a timing result.

For the original wide Attention geometry B512/D2048/480 attention nodes and
KV128, the removed proposal tensors total about480 GiB across state owners.
That is a static allocation delta, not a full-size measured peak or proof of
fit. Dense live KV, retained journals and backward buffers remain scale limits.
This change alone does not close full-size training or the F6 matrix.

Records are under `TASK/runs/fiber-append-*-clean01`; the allocation baseline
is `TASK/runs/allocator-resident02`, where
`TASK=/mi/data2T/zlong/tide-execution-flows`. Raw snapshots remain task-local.
The earlier `allocator-resident01` failed before construction because the
launcher passed a plugin file instead of its build directory; that failure is
retained and excluded from successful evidence. No implementation/gate failure
or unobserved full-size result is relabeled as a pass.
