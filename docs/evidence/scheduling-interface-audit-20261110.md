# Scheduling and batch-interface review — 2026-11-10

The reviewed source is `06b242eef2a203a950e30ced9b50d5c6d1cb8a73`.
The default host component scheduler can deliver a whole legal region window
across samples and logical times. Numeric preparation, Read, eligible regional
selection and Full contain real batches. This does **not** make every training
transpose, record publication or state lifetime a B×L operation.

One actual capacity defect and two bounded sources of repeated work were found
and patched after this baseline. The native/standalone patches require a new
build qualification; the baseline evidence below deliberately retains failures.
No new throughput claim follows from this audit.

## Review scope

The [hash-bound claims](scheduling-interface-audit-20261110-reviews.json) cover
42 files: 24 Python files with 110 AST-extracted functions, and 18 C++/header
files read as source. Twenty-one explicitly selected C++ function ranges have
their own line ranges and byte hashes under `manual_symbols`; this is manual
scope identity, **not** a complete compiler enumeration. The inventory remains
honest that C++ parsing was not run for the whole repository.

At this revision the automatic inventory selects 2,047 files and 4,649 Python
functions. Only the 42 files above are reviewed by this report. Its source
manifest hash is `2405a426616b4517b2446f311429b270b991a6316754b33a002807906b10eeeb`.
The earlier public-boundary audit is at another revision; its counts are not
silently accumulated into this one.

| Semantic layer | Actual files reviewed |
| --- | --- |
| Online certificate and component windows | Python `greedy`, `greedy_plan`, `window_plan`; native `greedy`, `greedy_plan`, `component_plan`, `component_window` |
| Block preparation and state commit | Python `blocks`, `block_policy`, `block_steps`, `packing`, `state_preparation`, `next`; native `block`, `block_prepare`, `block_step`, `state_preparation`, `stream_support`; public `block.h`, `kernel.h` |
| Local batching boundaries | Python Aggregate sharing, Read/Read VJP, Full, State batch/VJP, diagonal scan, weighted execution/Full/KV preparation/parameter rebind; native State VJP and weighted Full batch |
| State, transport and returns | Python `records`, `history`, `transfer`, `experiment/returns`; native delivery/communication and standalone return packing/check |

The manifest lists exact paths and individual scopes. It does not claim a full
resident scheduler audit, every local mathematical function, or the entire
streaming executor. The current backlog remains [ROADMAP](../ROADMAP.md).

## Confirmed capacity defect

`cpp/src/block.cpp::evaluate_block` moves and erases consumed fibers during
preparation when `compact_events=true`. Baseline
`component_window.cpp::Greedy::run_window` later subtracts `fibers.erase(...)`
from its global live count. That second erase returns zero, so consumed work
continues to count against simultaneous capacity.

The built `06b242e` native library gives this CPU FP64/no-grad witness: a
six-node chain, one sample and one input, `window`, logical stop 8. At every
point there is only one actual live fiber.

| compact_events | max_events | Baseline observation |
| --- | ---: | --- |
| false | 1 | completes, 6 candidates, `max_live_fibers=1` |
| true | 1 | wrongly rejects with `window live-fiber capacity exceeded` |
| true | 16 | completes, wrongly reports `max_live_fibers=6` |

The patch subtracts the component container's size decrease across preparation
and ordinary erasure, exactly once and before new delivery. This covers both
ownership paths while retaining future fibers in other components. It does not
raise capacity, remove pending messages or change the mathematical schedule.
The weighted eager default does not enable `compact_events`, so this witness
must not be presented as an explanation of the default FC16 timing.

`test_component_window_capacity.py` checks acyclic and feedback progression,
traced/untraced returns, independent scalar results, future component fibers and
parallel physical edges. Against the original built `06b242e`, the five ordinary
cases pass and five compact cases fail as predicted. Post-patch native success
requires a subsequent immutable build and is not inferred from the source edit.

## Repeated work removed

| Boundary | Baseline mechanism | Change and check |
| --- | --- | --- |
| Python node sequence lookup | Each node scans all sample-node sequence entries. For N nodes and S sequences, lookup work is O(N×S), or O(B×N²) in a dense region. | Build the node index once and preserve first-occurrence order. Existing independent multi-node/nonaligned/feedback/weighted tests check complete observables and gradients. |
| Python causal frame lookup | When across-time selection is unavailable, each time scans every frame to form that time's regional batch: O(T×F), or O(B×L²) for regular windows. | Build time-to-frame references once, O(F); preserve the existing chronological loop and selection boundary. History-dependent selection/Next tests are included. |
| Packed return with one unique value repeated in several fields | The one-row group calls the scalar host-copy path for every occurrence, despite having established one unique Tensor identity. | Copy once, then give each occurrence independent CPU storage. Python literal tests check bytes, NaN/signed zero, int64 above 2^55, no input mutation and one copy boundary. |
| Standalone scalar/grouped return | The same repeated one-row transfer exists; scalar also clones the result of every accelerator-to-CPU copy. | Mirror the one-transfer rule; clone same-device CPU input, reuse newly owned cross-device CPU storage for the first occurrence. `weighted-return-check` now checks actual transfer counters and independent alias storage. New SDK/device execution remains separate. |

The return changes apply to explicit packed groups; the scalar policy still
provides independent storage for every occurrence. CPU tests establish copy
ownership and byte semantics, not accelerator bandwidth. The index changes
remove host scans; they do not reduce the required attention work or establish
an end-to-end speedup by themselves.

## Interfaces and mature mechanisms

- Region SCC analysis caches static topology; current fibers are indexed by
  component. Acyclic components drain sealed windows, cyclic components use
  closure bounds from actual queued seeds. No CPU numerical trajectory supplies
  the candidate schedule. Positive delay and int64 bounds remain explicit.
- Local sequence legality depends on state adoption, clear policy and Next,
  not just graph type. History-independent regions may batch across time only
  after every descriptor is ready. Genuine feedback/control dependence can
  require smaller causal batches.
- Aggregate sharing and State preparation use actual input/parameter identity.
  KV preparation groups both samples and times, bounds temporary bytes and
  avoids copying parameter banks. Read/Full batches and bounded remote
  `copy_rows` are already real optimizations, not merely renamed scalar loops.
- Generic `state_vjp.bind_sequence` and native `state_sequence_vjp` still bind
  one depth at a time. Their structural signatures include exact time,
  preceding time and observation count, and each private graph publishes
  cloned row fields. Outside an applicable hidden-sequence VJP, a single
  numeric sequence call therefore still produces depth-wise training work.
  Kernel-specific sequence adjoints and finer capability keys are independent
  follow-ups; this is not evidence that those current gradients are incorrect.
- `PackedSequence.contents` is a tensor, but `views`, states, prepared operands,
  `FullInput`/`FullResult` and emitted slots remain row objects. Consumers often
  unpack then stack these fields. A future block interface should carry owners,
  offsets, exact times, field tensors and declared gradient roots through local
  boundaries, with explicit complete observation materialization. It must keep
  Read/Full snapshots, physical edges and exact continuation; this report does
  not implement that larger redesign.
- `Continuation.fork` copies metadata and borrows tensor graphs; `detach` starts
  a new autograd segment without making every tensor an independent snapshot.
  Complete experiment returns deliberately own independent CPU storage. These
  different lifecycle contracts should not be collapsed into an implicit clone
  at every internal boundary.

These observations identify interfaces for later work, not another live task
ledger. Regional numerical internals and the recurrent sequence proposal have
separate reviews; do not infer their status from this schedule audit.

## Reproduction and checks

```bash
python scripts/review_inventory.py --revision 06b242eef2a203a950e30ced9b50d5c6d1cb8a73 \
  --reviews docs/evidence/scheduling-interface-audit-20261110-reviews.json \
  --output artifacts/implementation-audit/scheduling-reproduced.json
```

The report/manifest regenerate the ignored inventory; no retired run directory
is required. Build the intended fixed revision using [development](../development.md)
and run the named regression files against that build. Ordinary CPU development
checks used the project-private Torch environment with vendor autoload disabled.

| Check | Observation |
| --- | --- |
| New capacity tests against the original `06b242e` native build | 5 passed / 5 failed, 0.80 s; the compact failures are the retained defect witness |
| Python component window/contracts/five memories, FP64 | 24 passed, 25 non-Python cases deselected, 36.19 s; includes independent forward/VJP/training/restore scopes of those tests |
| Python weighted return packing | 11 passed, 1 skipped, 6.67 s; standalone case requires `TIDE_WEIGHTED_BENCHMARK` and was not supplied |
| Updated native capacity and standalone return code | Await next fixed build/qualification; not claimed passed here |

The last two Python groups are working-tree development checks after the small
changes, not fixed-source performance qualification. No GPU/NPU test, heavy
benchmark or resident qualification was launched by this review.
