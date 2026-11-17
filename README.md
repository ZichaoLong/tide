# Tide instances

This branch implements the current instances from the fixed Tide mathematics at
`4f5b9f20a4030d68ee66ae2d26e34659a5e37234`.

The immediate product is efficient, trainable FC and HB instance implementations.
Semantic correctness comes first, performance second. General graph execution,
legacy APIs, old teaching bindings and LH-DAG/PDG are outside this delivery.
No retired model implementation or test is reused.

- [Current work](docs/STATUS.md) / [remaining work](docs/ROADMAP.md)
- [Use/configure/train/continue](docs/instances.md) / [current navigation](docs/README.md)
- [Scope and mathematical choices](docs/semantics.md)
- [Development](docs/development.md) / [execution](docs/execution-flows.md)
- [CPU/NPU FP32 results and limits](docs/evidence/instances-fp32-20261118.md)
- [Current performance lessons](docs/performance-lessons-instances.md)
- [Performance experience](docs/performance-lessons.md): historical observations,
  not compatibility requirements or qualifications of the new models.
- [Fixed mathematical source](math/instances/backbone-comparison.md)

The former implementation is retained in Git at `9afe84773afb72214e80b615cc704667945f0ccb`.
The new package is `tide-instances`, imported as `tide`; old consumers must migrate.
