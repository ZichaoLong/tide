# Architecture and navigation

## Layers

| Path | Responsibility |
| --- | --- |
| `python/tidegraph/` | Independent readable Torch oracle, fixtures and validation |
| `cpp/include/tide/` | Public graph, state, kernel and executor interfaces |
| `cpp/src/` | LibTorch kernels, streaming/frontier execution and binding adapter |
| `tests/` | Analytic formulas, invariant tests and differential qualification |
| `scripts/` | Configure/build/verify, durable job and re-entry helpers |
| `docs/semantics.md` | Adopted semantics and local formula/gradient contracts |
| `docs/ROADMAP.md` | Full implementation backlog and milestone gates |
| `docs/STATUS.md` | Only current handoff, next action and active jobs |
| `docs/evidence/` | Small immutable-source qualification reports |
| `docs/module-extension-plan.md` | Concrete next kernel/state interface work |
| `docs/attention.md` | Event GQA/window semantics, ragged cache and packed sequences |
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
in `packed.h`/`packed.cpp`. General region/Agg/Full program interfaces remain future
work; see the module extension plan. Slot tensors are included in all state
comparisons, resets, explicit detach and checkpoint v3.

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
