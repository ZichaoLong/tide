# Logical source domains and physical aliases

`SourceDomain` names incoming logical slots independently of the bijective
physical `PortLayout`. It supplies `edge_target[e]` and `input[p]`, indexed by
physical IDs. Every node's union of logical slots must be exactly `0..S-1`;
several physical wires/ports may name the same slot. Empty incoming domains have
S=0. Missing arrays, holes, negative/non-int64 slots are rejected before running.

The omitted domain resolves to the incoming physical port slots. An explicit
equivalent has the same graph identity. The domain is immutable graph structure,
not part of a shared parameter module. Python exposes `Graph.domain` and cached
`source_counts`; native compile resolves `source_domain` and builds counts once.
Physical CSR/CSC, input ledgers, message IDs and outgoing delivery stay unchanged.
After native topology changes, supply the new mapping or reset `source_domain`
as well as the physical layout before recompilation.

## Local program contract

Aggregate's `slots`, `SourceInput.slot` and contribution-map keys use logical
slots. Weighted/softmax Aggregate parameters and learned fiber-attention pooling
vectors have logical size S. All-source softmax normalizes each logical source
once, including absent logical sources; extra physical alternatives do not add
denominator terms. Active softmax and mean count only present logical sources.
Same-fiber attention orders KV/query rows by logical slot in both scalar and
packed paths, preserving cache order under physical-ID permutations.

A complete fiber must have at most one atom per logical source. Two physical
aliases arriving together fail before calling Aggregate, including zero-valued
atoms and boundary/edge collisions. The executor does not choose, merge or drop
one of them. A failed native cursor execution retains the existing restore-from-
snapshot recovery rule; source-domain collisions are execution failures.

Atom metadata and canonical public fiber order remain physical (subject to the
separate InputOrigin view). Receive/send scales are still physical parameters.
This mapping neither rewrites source tags nor implicitly ties weights. A specific
embedding must map tags if needed and explicitly share alternative-edge weights
and scales. Custom tag/order-sensitive programs therefore need an additional
projection proof. Mathematical source sums can retain the usual finite-precision
accumulation differences after physical reordering.

## Embedding, training and persistence

SettleGraph replaces boundary inputs with edges carrying the same logical domain;
its body programs and explicit parameter aliases remain shared. Adapter nodes
get independent domains. The existing canonical rank/stride clock is unchanged.

Checkpoint payload stays at the version in `semantics.md`; graph identity changes
and rejects earlier/different structural fingerprints before loading weights.
Checkpoint alias validation still requires clients to reconstruct parameter
sharing explicitly. In-memory continuations and isolated-root VJPs retain their
existing contracts; no extra replay is added to inference.

Tests compare a graph with unsplit wires against deliberately reordered,
phase-exclusive aliases, using independent reference/frontier/native schedules.
They cover absent sources, present zeros, learned pooling, complete state/cache,
messages/pending, every-cut cyclic continuation, isolated gradients, optimizer
updates, alias-aware save/resume and SettleGraph embedding in FP64/FP32.
This is a prerequisite for `lh-iocortex-plan.md`, not a single-PDG clock embedding.

Code: `source_domain.py` / `source_domain.h` / `source_domain.cpp`, Aggregate
request construction and graph validation. Tests: `source_domain_cases.py`,
`test_source_domain_contract.py`, `test_source_domain_schedules.py`.
