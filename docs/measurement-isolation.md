# Bounded measurement groups

`scripts/measurement_lanes.py` and `scripts/measurement_group.py` provide an
optional experiment-control boundary. They do not change graph execution,
model inputs, precision, physical chunking or optimizer semantics. The caller
supplies commands, environments, CPU masks, memory-node sets, RSS caps and
timeouts; accelerator leases and the common measurement lock remain caller-owned.
Site-specific masks, paths and NPU counts belong in external launch configuration.

Admission applies the existing half-resource policy once to the whole group:

- Lane CPU masks and NUMA memory-node sets must be disjoint. CPUs must belong
  to their selected nodes and the controller's effective affinity/cpuset.
- The sum of allowed CPU cores, including two reserved coordinator slots,
  must fit the aggregate CPU budget. Nominal ATen thread counts are not used
  as substitutes for a bounded CPU mask.
- Lane memory caps plus one explicit shared reserve must fit the aggregate
  host-memory budget. Each lane also must fit its selected-node estimate.
- The node estimate is `MemFree + 0.75 * max(0, Inactive(file)-Dirty-Writeback)
  - 2 GiB`, clipped to `[0, MemTotal]`. Active file pages and slab are excluded.
  This is a conservative admission heuristic, not a kernel MemAvailable value
  or a guarantee against unrelated memory pressure.

Children start under `numactl --physcpubind --membind`; unavailable binding fails
instead of silently using the whole host. Every live thread's CPU mask and
private anonymous page placement are sampled. Shared file-backed pages are
excluded from the NUMA assertion because binding does not relocate shared
library or existing page-cache pages. CPU masks separate cooperating children;
they do not reserve cores exclusively against other users.

The controller samples aggregate and per-lane RSS, enforces finite time/RSS
bounds and records commands, admission observations, placement and raw samples.
Sampling is host-side and adds declared overhead; it never synchronizes device
tensors. Both solo and overlap controls must use the same monitor policy.
On any failure or cancellation, it terminates and reaps only its owned process
groups, including adopted grandchildren. Failed/partial records remain failed;
the caller must not count cancelled companions as completed measurements.

No performance isolation is inferred from disjoint masks alone. Qualify each
intended workload class with matched solo/overlap controls, using the same
source, software stack, card placement, thread pools and memory policy. Keep
bound/concurrent series distinct from earlier unbound serial results. A finite
pilot can screen interference, but does not replace the three independent
processes required for a recommendation. Profiling remains separate.

Dependency-free checks (including actual NUMA child processes and cleanup):

```bash
python -m unittest discover -s tests -p 'test_measurement_lanes.py' -v
```

The active qualification, exact budgets, source identities and observed support
belong in [STATUS](STATUS.md). This implementation alone does not qualify a
parallel full-size workload or change any historical benchmark result.
