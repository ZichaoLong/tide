# Stable local program ports

Node parameter modules may be shared, and graph embeddings may turn external
ports into edges. Local programs therefore address numbered input/output slots
owned by a node, rather than embedding physical graph IDs in their weights.
The mapping belongs to graph structure and is protected by graph identity.

`PortLayout` has four arrays indexed by physical IDs:

| Array | Local slot of |
| --- | --- |
| `edge_source[e]` | outgoing edge e at its source node |
| `edge_target[e]` | incoming edge e at its target node |
| `input[p]` | external input p at its target node |
| `output[p]` | external output p at its source node |

Incoming and outgoing slots are separate namespaces. Each node/direction must
be a bijection onto `0..degree-1`. Parallel edges remain distinct; boundary
ports and edges may share a node but cannot share a slot in the same direction.
All slots are nonnegative int64 values. Missing arrays, duplicates and holes
are rejected at graph construction/compilation, before execution.

The automatic incoming order is external input ports followed by incoming
edges, each in physical ID order. The automatic outgoing order is edges then
external output ports. Explicit mappings may permute either order. A default
layout and its explicit equivalent have the same structural identity.

Python `Graph.ports` resolves the optional `layout`, and `port_indexes` caches
the inverse indexes. C++ `Graph::compile` resolves an absent optional layout
in place. Both build flat `PortIndex` rows with node offsets and `(kind,id)`
bindings in local-slot order; kind 0 denotes a boundary port, kind 1 an edge.
An outgoing local slot resolves in O(1) as `bindings[offsets[node] + slot]`.
The existing physical CSR/CSC indexes continue to retain physical edge IDs.
Compilation/storage is O(nodes + edges + boundary ports); no per-event graph
scan is needed for slot lookup. These properties are not throughput evidence.

Python `dataclasses.replace` on an automatic graph recomputes its layout.
After modifying an already compiled native topology, either supply a compatible
layout or reset `graph.layout` to `std::nullopt` before compiling. Compilation
does not silently reinterpret an explicit layout. Graphs held by executors are
immutable copies.

## SettleGraph and persistence

SettleGraph embedding keeps every body node's local slots. Original input-port
slots become target slots of the source adapter's edges; output-port slots become
source slots of edges to the readout adapter. Body parameter modules retain their
sharing and do not contain these physical mappings. Projection restores source
tags and re-sorts mixed boundary/internal fibers to the original canonical order.

Layouts participate in Python's graph fingerprint and native structural identity.
Current graph/checkpoint versions are in `semantics.md`. Checkpoints
with a different graph fingerprint, including old fingerprints without layouts,
are rejected before weights change. There is no implicit checkpoint migration.

Full/Emit now consumes local output slots; physical send/output scales remain
outside its program. Source-aware Aggregate is described in `aggregate-programs.md`.
Program numerical contracts and qualification are separate from layout indexing.

Implementation: `python/tidegraph/ports.py`, `cpp/include/tide/ports.h`,
`cpp/src/ports.cpp`, graph definitions and `SettleGraph.embed/project`.
Tests: `tests/test_port_layout.py` (parallel edges, explicit permutations,
malformed layouts, graph/checkpoint guards, sharing and mixed-input embedding).
