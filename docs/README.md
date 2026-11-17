# Current navigation

The current implementation is the instance rewrite at `src/tide`, with tests in
`tests`, comparison programs in `benchmarks`, configurations in `configs`, and
fixed mathematics in `math`.

- [Use/configure/train/continue](instances.md)
- [Current work](STATUS.md) and [remaining work](ROADMAP.md)
- [Scope](semantics.md), [code navigation](architecture.md), [mathematical entry points](semantic-map.md)
- [Development](development.md) and [execution](execution-flows.md)
- [FP32 implementation, validation and measurements](evidence/instances-fp32-20261118.md)
- [Current measured performance lessons](performance-lessons-instances.md)
- [Historical performance lessons](performance-lessons.md)

Other pre-rewrite documents and evidence retain their original source identities
for historical reference. They are not requirements, public interfaces or
qualification of this rewrite. In particular the old generic graph, resident,
mixed-a/b/c and backend matrices do not define the new delivery. Consult them
only for a relevant lesson; do not recover or reuse the retired implementation.
