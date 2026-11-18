# Tide instances

Efficient, trainable implementations of selected Tide FC/HB bindings. The active
mathematical revision is `4f5b9f20a4030d68ee66ae2d26e34659a5e37234` in `math/`.
Semantic correctness comes first, performance second.

The instance product is a continuing route; the general graph foundation is a
separate future direction. [Project scope](docs/project-scope.md) records that
agreement and how semantic cycles change.

- [Current status](docs/STATUS.md) / [current backlog](docs/ROADMAP.md)
- [Use, train and continue](docs/instances.md) / [clean development](docs/development.md)
- [Named training presets and mode-specific recommendations](docs/execution-recommendations.md) (FP32 first for training)
- [Throughput by scale/topology and the mathematical computation flow](docs/computation-and-throughput.md)
- [Documentation map](docs/README.md) / [architecture](docs/architecture.md)
- [Adopted semantics](docs/semantics.md) / [precision](docs/precision.md)
- [Four joint Transformer comparisons](docs/benchmark-contract.md)
- [Source-scoped numerical, performance and continuation evidence](docs/evidence/README.md)
- [Applicable performance experience](docs/performance-lessons.md)

The package is `tide-instances`, imported as `tide`. CPU/NPU FP32 and explicit
BF16 paths have been exercised, including training and restored updates. The
linked evidence retains finite numerical deviations and cross-precision support
differences; completed execution does not imply identical optimization trajectories.
Retired implementation cycles are available only through explicit [historical lookup](docs/history.md).
