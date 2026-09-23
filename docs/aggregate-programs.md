# Source-aware Aggregate programs

Built-ins and extension seams are [qualified](evidence/aggregate-programs.md).
The source-origin adapter below fixes the subsequent tag-sensitive embedding
counterexample and is [qualified](evidence/source-origins.md).
Aggregate receives a complete nonempty source-tagged fiber, logical time, stable
local input slots and legacy physical source scales. Programs are functional,
immutable during execution, and independent across events. Source tags and input
positions remain available; logical time is not a token position.
Slots now resolve through `SourceDomain` (`source-domains.md`). Its default
is the incoming physical layout; explicit exclusive aliases share a logical
coefficient. Duplicate logical arrivals are rejected before invoking the program.

Return a summary tensor and an optional sparse map of per-source contributions,
indexed by present local input slots. The summary need not be a sum for a custom
program. Contributions preserve information for future source-aware state
programs; built-in memory profiles consume the summary. The raw fiber remains in
the event. Public traces expose `contributions` as a local-slot map. Empty fibers
never call Aggregate. A present zero stays present.

## Built-in formulas

Let A be the present input slots, D the static input domain, and y_j=s_j*x_j,
where s_j is the existing physical input/edge scale. Fold in canonical atom order:

| Profile | Per-source contribution c_j; summary is sum over A |
| --- | --- |
| `sum` | y_j (the original profile) |
| `mean` | y_j / number of present sources |
| `weighted_mean` | softplus(m_j) * y_j / sum over A of softplus(m_k) |
| `active_softmax` | exp(l_j) * y_j / sum over A of exp(l_k) |
| `all_softmax` | exp(l_j) * y_j / sum over D of exp(l_k) |

Softmax uses the stable Torch implementation. Positive masses avoid a signed or
zero denominator; numerically unrepresentable zero total mass fails explicitly.
Each mass/logit is an independent local-slot parameter. Absent-source parameters
are disconnected for weighted mean and active softmax. All-source logits are
connected through the denominator even when their messages are absent. Physical
source scales are connected only for present messages in every profile.

Node-local programs contain no physical edge/port IDs. SettleGraph remapping
preserves local slots, programs and parameters. Identity boundary adapters use
`sum`. Profile and local layout participate in graph/checkpoint identity.

## Packing and acceptance

The built-in batch implementation groups identical source-slot patterns without
padding absent sources. Local semantic replay preserves first-order public-root
VJPs. Scalar custom programs are a visible fallback. The all-source denominator
requires inspecting the static domain; active-source profiles visit only present
inputs during execution. No speed claim follows from these implementations.

Acceptance: analytic forward/VJP including absent versus zero and singleton
normalization; Python/native serial/parallel/packed streaming and frontier;
independent topology schedules; mixed external/internal SettleGraph fibers and
sharing; cursor/cuts/checkpoint; isolated contributions/output roots and optimizer
behavior. A native standalone custom program must exercise tags, time, slots and
autograd without Python callbacks. Invalid results and unsupported adapter
programs fail explicitly.

## Extension and navigation

Optional native Streaming [packed source transport](packed-transport.md) shares
scaled source rows with attention and retains a packed summary between stages.
It preserves the same per-event content/trace and first-order VJP contract.

Native clients provide `NodeWeights::aggregate_kernel` implementing
`AggregateKernel` in `cpp/include/tide/aggregate.h`. Source atom metadata is copied;
Tensor storage remains shared and read-only. Python programs are
parameter-owning `AggregateProgram` modules, passed in `Model(aggregate_programs=)`
and carrying a versioned profile matching `Node.aggregation`. A Python-only
program cannot cross the native adapter without a corresponding native program.

`aggregate_kernel.cpp` holds built-ins; `aggregate_evaluate.cpp` handles validation
and replay. Python's independent counterpart is `aggregate.py`. Scalar fallback,
packed Aggregate calls and semantic replays have separate statistics. Current
graph/checkpoint versions and identity rules are in `semantics.md`.
`cpp/test/custom_aggregate.cpp` checks a time/tag/slot-dependent program with
hand-computed loss 136, input gradients 52 and 4, gain gradient 116 and a
disconnected sample. Related tests live in `test_aggregate_{formulas,schedules,contract}.py`.

## Source-origin views under embedding

A physical boundary edge is not an external atom. Its source ID and send time
also differ from the original port ID and input position. Custom programs can
observe these differences, so preserving only local slots is insufficient.

`Graph.origins` contains explicit `InputOrigin(edge, port, stride)` entries.
Aggregate input construction first resolves the physical scale and local slot,
then presents that edge's atom as `(kind=external, source=port,
position=send_time/stride)`. Arrival time, target, sample and payload are preserved.
The declared send-time lattice is checked; no rounding is allowed. Sources are
sorted into canonical program-visible order after projection, including mixed
external/internal fibers. Raw routing, pending records and traces keep physical
identities; SettleGraph's public projection restores its original fiber records.

The mapping is graph-owned, included in graph/checkpoint identity and independent
of shared parameters. Native compilation stores a flat edge-to-origin index;
execution visits only the present fiber. Graphs without origin views allocate no
origin index and skip the extra source sort. SettleGraph supplies the mapping for
each boundary input adapter automatically. Built-in and custom Aggregate programs
then see the original source domain without retaining physical IDs in modules.
Aggregate's result now carries this complete source-aware content through state,
Read and Full; see `content-programs.md` for the implementation and qualification
status of that interface extension.
