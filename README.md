# Tide instances

Efficient, trainable implementations of selected Tide FC/HB bindings. The active
mathematical revision is `4f5b9f20a4030d68ee66ae2d26e34659a5e37234` in `math/`.
Semantic correctness comes first, performance second.

The instance product is a continuing route; the general graph foundation is a
separate future direction. [Project scope](docs/project-scope.md) records that
agreement and how semantic cycles change.

- [Current status](docs/STATUS.md) / [next work programme](docs/ROADMAP.md)
- [Use, train and continue](docs/instances.md) / [clean development](docs/development.md)
- [Documentation map](docs/README.md) / [architecture](docs/architecture.md)
- [Adopted semantics](docs/semantics.md) / [precision](docs/precision.md)
- [Four joint Transformer comparisons](docs/benchmark-contract.md)
- [Measured FP32 results](docs/evidence/instances-fp32-20261118.md)
- [Applicable performance experience](docs/performance-lessons.md)

The package is `tide-instances`, imported as `tide`. FP32 CPU/single-NPU evidence
exists; BF16 is an agreed next delivery target, not yet a qualification claim.
Retired implementation cycles are available only through explicit [historical lookup](docs/history.md).
