# Architecture and navigation

## Layers

| Path | Responsibility |
| --- | --- |
| `python/tidegraph/` | Independent readable Torch oracle, fixtures and validation |
| `cpp/include/tide/` | Public graph, state, kernel and executor interfaces |
| `cpp/src/` | LibTorch kernels, streaming/frontier execution and binding adapter |
| `tests/` | Analytic formulas, invariant tests and differential qualification |
| `scripts/` | Configure/build/verify, durable job and re-entry helpers |
| `docs/verification.md` | Explicit development/full gates and frozen worktree lifecycle |
| `docs/semantics.md` | Adopted semantics and local formula/gradient contracts |
| `docs/ROADMAP.md` | Full implementation backlog and milestone gates |
| `docs/STATUS.md` | Only current handoff, next action and active jobs |
| `docs/evidence/` | Small immutable-source qualification reports |
| `docs/module-extension-plan.md` | Local program catalog and extension contracts |
| `docs/attention.md` | Event GQA/window semantics, ragged cache and packed sequences |
| `docs/streaming-benchmark.md` | Bounded native workload, parity anchors, timed phases and experiment records |
| `docs/streaming-cursor.md` | Native owned state, incremental advance and explicit snapshots |
| `docs/checkpoint-ownership.md` | Named parameter/optimizer ownership and preflight value restore |
| `docs/packed-autograd.md` | Packed values, independent gradient connectivity and replay cost |
| `docs/local-ports.md` | Stable node slots, physical wire mappings and compact inverse indexes |
| `docs/source-domains.md` | Logical incoming slots, exclusive physical aliases and denominator/cache correspondence |
| `docs/state-clocks.md` | Compact periodic state clocks, global continuation and delegated batching |
| `docs/full-programs.md` | Full program API, per-slot emission, absence and replay contracts |
| `docs/aggregate-programs.md` | Tagged fibers, source contributions, normalized profiles and program API |
| `docs/read-programs.md` | Region Read modes, independent readout kernels and scalar VJP validation |
| `docs/content-programs.md` | Complete-content views, state extensions, packing and replay |
| `docs/next-programs.md` | Next requests, clear policy, state validation and prefill gates |
| `docs/region-programs.md` | Selector interfaces, typed history, controls and persistence |
| `docs/lh-add-plan.md` | Lazy Add design notes, decay precision and physical state projection |
| `docs/lazy-add.md` | Explicit tick-repeat Add, physical decode, batch buckets and parameter-epoch limits |
| `docs/lh-full.md` | LH activation/normalization backbones and per-edge signaling mapping |
| `docs/lh-attention-plan.md` | Same-fiber cache/visibility/decay and post-attention pooling gate |
| `docs/fiber-attention.md` | Same-fiber sum attention, encoded log bias and independent scalar anchors |
| `docs/fiber-packing.md` | Source/event offsets, per-sample attention buckets, visibility masks and replay |
| `docs/fiber-pooling.md` | Post-attention mean/linear/softmax Confluence, slot domains and vector VJPs |
| `docs/token-window.md` | Sealed body-to-token conversion, norm-only readout, labeled logits and continuation |
| `docs/lh-pronounce-plan.md` | Token clock, sealed-window adapter, original readout oracle and whole-model obligations |
| `docs/lh-iocortex-plan.md` | Actual whole-model oracle, four-block wiring and remaining containment obligations |
| `docs/single-graph-training.md` | Two-clock/single-PDG VJPs, shared optimizer owners and partial-window truncation |
| `docs/lh-single-graph.md` | Bounded single-PDG IOCortex construction, every-cut projection and readout ledger boundary |
| `docs/lh-iocortex.md` | Original whole-model entry points, precise graph projection and cross-language oracle scope |
| `docs/lh-selector.md` | Descriptor precision, LH selector profile and original C++ component oracle |
| `docs/lh-compatibility.md` | C++ LH inference mapping and unverified obligations |

One semantic spine: `SettleGraph -> encoded TimedDAG -> PositiveDelayGraph`.
Embeddings transform clocks, boundary nodes and source tags explicitly. Local
operators are shared within a language; schedules are independent. Cross-
language implementations and analytic examples check the operator formulas.

## Runtime decisions

- Graph compilation builds CSR outgoing and CSC incoming edge indexes once.
  Runtime touches only scheduled events, candidate regions and outgoing edges.
- A batch is independent sequence instances with disjoint state/history. Lazy
  state allocation is keyed by (sample, node); histories by (sample, region).
- Fibers contain tagged external atoms or edge messages. Sort deterministically
  before aggregation; retain source identity even when payloads are equal.
- Input completeness is explicit. A sealed half-open window carries all external
  records in that window. Queue emptiness alone never advances logical time.
- `Next` and comparison snapshot precede `Full`. Full may be packed/parallel;
  it cannot write persistent state. Node and region owners are unique.
- Native workers compute independent node results. The coordinating thread
  commits them in stable order; workers inherit Torch thread-local state.
- TimedDAG frontier blocks require declared exact contracts. Stateful selection,
  clear/reset and feedback may restrict batching. Report the block sizes and
  fallback, rather than naming a sequential loop “prefill”.
- Runtime checkpoints serialize values/spec identity, not a live autograd graph.
  In-memory cuts preserve graph connectivity unless explicitly detached.

State preparation now uses `cpp/include/tide/kernel.h`, with named tensor slots
and an opt-in exact sequence contract. Built-in profiles are EMA, identity and
selective diagonal SSM, Linear/Delta and event GQA/window attention. Packed state
prefill is isolated in `block_prepare.cpp`; its checked segment representation is
in `packed.h`/`packed.cpp`. Full programs and sparse per-slot delivery are separate
from scheduling. Source-aware Aggregate uses `aggregate.h`/`aggregate.py`, with
native kernels and replay evaluation in separate source files. Read is separate
in `readout.py`/`read.h`/`read.cpp`; Next in `next.py`/`next.h`/`next.cpp`.
Regions use `region.py`/`region.h`, with typed history in `history.py`/`types.h`.
Node and region tensor slots participate in comparisons, explicit detach and
the checkpoint schema in `semantics.md`. Built-in counters use checked int64 arithmetic.

Build qualification checks a source fingerprint and binary hashes from
`build/build-manifest.json`; stale native modules cannot certify newer C++ code.
Custom build directories propagate explicitly to pytest. `scripts/clean_artifacts.py`
defaults to a dry run and protects referenced, failed and active-job artifacts.

## Module boundaries

Nodes expose Agg, Upd, Read, Next, Full/Emit with typed state. State modules need
step and optional exact block interfaces; advertise limitations (mask, position
clock, same-fiber policy, window, reset). Initial profiles cover a small EMA and
FFN as a validation anchor; attention/GQA, linear attention, DeltaRule, SSM and
SwiGLU are separately qualified implementations. No generic interface by itself
proves that an arbitrary pretrained checkpoint can be imported.

Packed data needs sample IDs, node IDs, event/sequence offsets, source-edge IDs,
logical times, positions and state slots. Padding never creates candidates.
An absent message is distinct from a numerical zero message.
