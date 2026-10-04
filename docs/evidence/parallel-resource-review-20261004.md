# Parallel resource review and proposed execution lanes

Read-only host/launcher inspection on 2026-10-04, while the already running
formal Python CPU cell remained unchanged. No concurrent experiment, binding
change, NUMA allocation probe or new profile was launched.
[Snapshot and source identities](parallel-resource-review-20261004.json).
This is a proposal for alignment after the current evidence commit, not an
implemented resource manager or qualified parallel-performance result.

The host has 320 physical cores, four sockets, eight NUMA nodes of 40 cores
and about 2.01 TiB RAM. The snapshot found 1,446.48 GiB globally available,
no swap, and a current aggregate policy budget of 160 cores/723.24 GiB.
Availability varies; each launch must recheck it. NUMA nodes total about
248–252 GiB each. Per-node MemFree is not reclaimable capacity and must not
be treated as a precise available-memory budget.

`numactl` and `taskset` are installed. Current jobs permit CPU0–319 and memory
nodes0–7; their separate services and Nice10 do not partition cores, memory
bandwidth or cache. Neighbor NUMA pairs have distance15 and other pairs20
(local10). The 16 NPU PCI endpoints report NUMA=-1 and all CPUs local, so
this inspection does not establish NPU-to-host locality. NPU topology reports
SIO within the eight two-chip pairs and HCCS_SW between other chips; disjoint
cards would still share parts of the communication fabric.

The current primary matrix uses11 NPUs per accelerator case. Two unchanged
cases need22 and cannot coexist on16. An8+8 experiment would change card count,
placement, capacity and timing configuration; existing11-card results would
not qualify it. The five spare cards may fit separately qualified small cases,
but do not imply that another original-size training case fits.

The existing formal launcher takes a global `online-measurement.lock`, checks
one process against half the effective CPU/memory availability, and reserves
136 GiB for retained history/service headroom. It does not reserve the combined
cost of concurrent children or inspect their selected NUMA memory. Removing
the lock alone would leave these admission assumptions invalid. Its present
CPU guard requires at least66 budgeted cores, so arbitrarily assigning a small
CPU mask would fail admission even when ATen uses only16 threads.

Suggested first configuration after user authorization:

1. One CPU lane plus one11-NPU lane, each with a separate service, output and
   queue record. Use disjoint CPU masks and memory-node sets from process
   creation, retain bounded ATen/OpenBLAS/vendor pools, and record effective
   affinity and NUMA placement. A two-socket mask per lane preserves the current
   CPU admission guard; actual node choices and memory limits need qualification.
   These masks separate this project's children, not unrelated users' processes.
2. Account for combined host-memory demand, one shared reserve, selected-node
   capacity, NPU leases and construction peaks before admitting overlap.
   Preserve the protected historical process and the aggregate half-resource
   policy. Do not run large builds or trace exports beside formal measurements.
3. Start with one bounded Add CPU/resident pair. Compare each case running alone
   against overlap under the same binding and configuration. Check wall time,
   actual work, peak RSS/HBM and effective placement. Binding itself can alter
   performance, so old unbound timings are not the solo controls. Keep this
   qualification separate from the old formal series; do not silently pool them.
   A proposed tolerance is at most5% overlap slowdown for either lane; one pilot
   can screen this but does not replace the three fresh processes needed for a
   recommendation. If isolation fails, retain the result and schedule that class
   serially rather than begin an open-ended tuning sweep.
4. Pair by memory class instead of applying overlap to every case. Only after
   the first overlap is useful should a second CPU lane or small-NPU lane be
   considered. Preserve finite queues, bounds, failures and no automatic retry.

The following illustrates the existing conservative admission numbers, using
one shared136 GiB reserve. These are memory allowances, not measured total RSS
or proof of fit under NUMA binding:

| Pair | CPU estimate GiB | NPU process host allowance GiB | Combined including reserve GiB | Fits snapshot global723.24 GiB budget? |
| --- | ---: | ---: | ---: | --- |
| CPU Add training + resident | 191.27 | 280 | 607.27 | Yes globally; node/peak qualification pending |
| CPU Attention training + resident | 412.11 | 280 | 828.11 | No |
| CPU Attention training + mixed | 412.11 | 64 | 612.11 | Yes globally; node/peak qualification pending |

NPU HBM remains a separate per-device budget. Global fit alone cannot guarantee
local-node fit or freedom from external contention. The proposed scheduler must
avoid counting the same memory budget twice, while retaining rather than merely
subtracting measured safety allowances to make an overlap appear feasible.

No parallel speedup is measured or promised. One CPU lane and one NPU lane can
hide some work behind each other; they cannot halve a matrix dominated by NPU
cases. Remaining serial forecast sums are planning estimates from reduced-batch
pilots and cross-family transfers, not completion-time promises. The current
pause and the next authorized action are owned by [STATUS](../STATUS.md).
