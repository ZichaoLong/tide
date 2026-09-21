# Complete content and state-program extension

[Qualified](evidence/content-programs.md). This extends the content
interface; Read and Next are separate (`read-programs.md`, `next-programs.md`).

Python `Content` and native `ContentView` expose summary `value`, canonical
program-visible `sources` and optional per-slot `contributions`. Each source has
its stable local input slot, atom metadata/value and physical source scale.
Aggregate constructs this content once, after applying graph-owned origin views;
state step/Read, Full and packed state metadata use it unchanged. Raw physical
fibers stay separate for routing and public trace projection.

Native views borrow the event's source/contribution arrays only for synchronous
program calls. Never store those views in State or mutate shared tensor storage.
Persistent memory belongs in named tensor slots. Python's content records have
the same functional contract. Local summary-only tests can construct Content
with no source metadata; actual graph events always have nonempty sources.

## State and batch APIs

Native StateKernel step and the independent ReadKernel receive ContentView. Independent batch and
sequence APIs retain packed values/times with aligned ContentViews. The default
fallback replaces each view's summary with its packed row before invoking step;
source metadata and contributions still correspond to that event. Packed inputs
validate view count and summary shape/device/dtype.

Python custom kernels subclass parameter-owning `StateProgram`, declare a
versioned profile matching `Node.memory` and enter via `Model(state_programs=)`.
The native adapter rejects Python custom programs and overridden built-ins
without a corresponding native implementation. Built-in sharing also validates
memory kind and attention policy. State program parameters participate in
optimizer state, sharing and checkpoints through normal module registration.

Custom state programs default to no sequence contract. An exact scalar sequence
can opt in, but `joint_sequence` remains false unless it supplies actual joint
computation. Native `state_scalar_batch_steps` and both implementations'
`state_scalar_sequence_steps` distinguish these fallbacks from API call counts.
They do not imply any performance result. Built-in EMA, SSM, Linear/Delta and
attention preserve their existing numeric kernels and block contracts.

## Replay and checks

Semantic replay consumes complete original event content. Python separates
`propose` from `describe`, removing the previously discarded extra Read during
state replay. Packed numerical Read uses each event's correct previous state;
semantic Read consumes the bound proposal. First-order public-root scope,
None/connected-zero behavior and inference's no-replay rule remain unchanged.

The custom test state uses source slots, kind, origin position, physical scales
and Aggregate contributions. Its Read consumes proposal plus one contribution;
custom Full uses comparison plus source tags and a contribution. Independent
loss/VJP formulas cover selected clear, packed scalar fallbacks, native serial/
parallel/streaming/frontier, boundary origin encoding and isolated samples.
Python additionally covers direct/encoded SettleGraph, module registration and
optimizer/checkpoint state. See `tests/test_content_programs.py` and the standalone
`cpp/test/content_programs.cpp` checks linked into `tidegraph-kernel-check`.
