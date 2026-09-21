# LH selector profile and descriptor precision

`norm-fp64-v1` is a Read profile computing the L2 norm of the value visible in
the region's content/old/proposal mode. The norm accumulates in FP64 even for
FP32 payloads; its result is a finite scalar FP64 tensor. Casting an already
rounded FP32 norm is not this operation. Zero norm uses Torch's connected-zero
VJP. Its ordinary norm VJP is a Tide training choice; LH is an inference oracle
only and its own selector detaches scores.

Read programs declare `precision = payload | float64`; native ReadKernel exposes
`descriptor_dtype(payload_dtype)`. The profile name is part of graph identity.
The adapter rejects changes to built-in precision policies. No graph schema bump
is needed beyond v11 because new profiles have new explicit names.

Region requests now include payload dtype/device metadata separately from
descriptors. A region can contain both payload-precision and FP64 descriptors.
Count, positive, tensor-history and LH profiles stack/promote scores, calculate
softmax in that score dtype and explicitly convert controls to payload dtype.
Tensor-history sums descriptors in their promoted dtype, converts the sum to
payload dtype and then updates its payload-dtype memory. Casts retain ordinary
Torch gradients. Custom programs remain responsible for the declared metadata
of their controls and history.

`lh-count-affect-v1` stores `node_maps['selected']` and `node_maps['affected']`.
For a complete nonempty candidate set, compare prior selected count ascending,
prior affected count descending, descriptor descending and canonical ID ascending.
Increment affected for every candidate and selected for active nodes; both are
checked nonnegative int64 counts. Empty regions do nothing. This profile requires
`count_priority=True`. It has no learned parameters. Its softmax controls are
Tide's explicit training/control extension; LH itself returns selected signals.
Use full-capacity separate regions for forced-activity hubs.

## Original-source comparison

The optional oracle compiles the actual original LH `Selector.cpp`, `Adjacency.cpp`,
`GraphConfig.cpp` and `BaseConfig.cpp` from a checksummed snapshot. It does not
rewrite their algorithms or use LH's Python interpreter. Full C++ sources/headers
and the required nlohmann headers are snapshotted for reproducibility. Build flags
select CPU, runtime assertions and serial LH execution. Tide itself is checked
in serial, node-parallel packed streaming and frontier modes.

```sh
python scripts/snapshot_lh.py --source /path/to/lh --json-include /path/to/include \
  --output-dir artifacts/lh-source-unique
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python scripts/build.py --jobs 2
python scripts/check_lh_selector.py --device cpu --dtype both \
  --snapshot artifacts/lh-source-unique --output-dir artifacts/lh-oracle-unique --jobs 2
```

Use the durable job workflow for builds. The oracle verifies snapshot identity,
current source fingerprint and the core static library's hash before running,
and rechecks them afterward. It records its CMake/binary identity and per-dtype
commands/results. Paths must be new; the reference repository is read-only.

Each dtype covers 12 cases: lead-point inclusion on/off, budgets 1/2, and three
Tide schedules. Each case has 24 ticks and four ragged samples, dense score ties,
whole idle ticks, empty samples, sparse node IDs, forced activity and FP64 norm
distinctions. Compare selected IDs and payloads exactly, and both history maps,
at every tick against **both** original heap and tensor implementations.

LH stores counters in int32 and its tensor path builds composite double ranks.
These small-count cases stay below counter overflow and composite-score precision
limits. Equality outside that domain is not claimed. Tide preserves its checked
int64 policy rather than reproducing overflow. This is a selector component
comparison, not whole-LH inference parity: Add idle decay, same-fiber attention,
edge signaling and Pronounce still require complete state/message/clock mapping.
Clean FP64/FP32 qualification is recorded in [evidence](evidence/lh-selector.md).
