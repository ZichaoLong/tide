# Current evidence

The Reception, FP32 training and resident-inference cycle is recorded in:

- [Starting cost and SSD measurements](reception-start-20261118.md).
- [Three Reception profiles, numerical deviations and actual restored updates](reception-numerics-20261118.md).
- [SSD/Reception candidates, memory tradeoffs and two-sided device cost attribution](reception-performance-20261118.md).
- [One hundred FP32 updates, capacity and joint A–D inference/grad-forward/training](reception-training-20261118.md).
- [Resident large-model inference, regional execution and uninterrupted streams](reception-inference-20261118.md).
- [Integrated CPU, installed wheel, secondary BF16 and placed-path qualification](reception-validation-20261118.md).

Each report keeps its actual source and workload. Finite numerical and support
deviations remain attached to the original comparisons.

The training-default, numerical-path and cost-attribution cycle is recorded in:

- [Finite budget searches, pure batching/recomputation and new training candidates](training-search-20261118.md).
- [Normal timing, saved memory, actual hardware traces and Transformer tuning](training-costs-20261118.md).
- [Fixed nonempty-state comparisons, repeated backward and first Attention difference](training-paths-20261118.md).
- [Twenty continuing updates, KV bounds and actual new-process restored updates](training-continuous-20261118.md).
- [Recommended-shape A–D training, inference/grad-forward and cached prefill/decode](training-modes-20261118.md).
- [Source-scoped CPU checks, new installed wheel and final handoff identity](training-validation-20261118.md).

Loadable choices and their numerical/memory limits are in
[execution recommendations](../execution-recommendations.md).

The capacity, throughput and regional-execution phase has concluded. Its records
retain actual frozen source identities, original numerical failures and bounded
search limits:

- [Integrated CPU suite, support catalog and final identity checks](capacity-validation-20261118.md).
- [Longer inputs, larger batches and all four joint Transformer matches](capacity-search-20261118.md).
- [Execution variants, memory lifetime and expanded-batch throughput](capacity-execution-20261118.md).
- [BF16 batching correction and bounded first-divergence witnesses](capacity-rounding-20261118.md).
- [0.5B/1B continuous execution and fresh-process restored-update timing](capacity-continuous-20261118.md).
- [Whole-region placement, two-NPU concurrency and 9B/17B capacity](region-capacity-20261118.md).
- [Standalone LibTorch CPU checks](native-regional-cpu-20261118.md) and
  [CPU/NPU regional performance, restored gradients and pool streams](native-regional-performance-20261118.md).
- [Isolated new wheel, optional bindings and bounded restored-gradient diagnosis](capacity-installed-20261118.md).

The preceding five-workstream records remain scoped to their own implementations:

- [Seven-point CPU/NPU four-class representative matrix](roadmap-representative-20261118.md).
- [CPU/NPU continuous use, complete timing and actual restored updates](roadmap-continuous-20261118.md).
- [Measured hotspot choices](roadmap-hotspots-20261118.md).
- [Three Attention families and actual operator witnesses](roadmap-attention-20261118.md).
- [Sparse, shape and useful-combination comparisons](roadmap-supplementary-20261118.md).
- [Actual 0.5B/1B four-class scale comparisons](roadmap-scale-small-20261118.md).
- [9B/17B scale and actual capacity](roadmap-scale-large-20261118.md), including completed CPU17B inference and two BF16 updates.
- [Multi-NPU placement, transport failures and corrected launch environment](roadmap-placement-20261118.md).
- [Initial precision and rounded-allocation diagnosis](roadmap-numerics-initial-20261118.md),
  then [continued VJP, repeated backward and restored updates](roadmap-numerics-followup-20261118.md).
- [Bounded whole-model capture boundary](roadmap-resident-20261118.md).
- [Clean installed wheel and actual restored updates](roadmap-installed-20261118.md).
- [Final artifact retirement and isolated post-cleanup smoke](roadmap-retirement-20261118.md).

These reports preserve both failures and passes; the final handoff and remaining
qualification boundaries are in [STATUS](../STATUS.md). Each links its reviewed JSON evidence.

- [FP32 instance delivery, 2026-11-18](instances-fp32-20261118.md), with
  [complete reviewed observations](instances-fp32-20261118.json).
- [Instance governance and project cleanup, 2026-11-18](governance-cleanup-20261118.md),
  with [preservation, deletion and installed-smoke records](governance-cleanup-20261118.json).
- [Follow-up removal of released downstream inputs and small remnants](cleanup-followup-20261118.md),
  with [exact retirement and preservation records](cleanup-followup-20261118.json).

Each report identifies source, configuration, actual tests and remaining limits.
[Main-branch file mappings](capacity-validation-20261118.md) make original sub-worktree
implementation/harness files reproducible from the delivered branch; whole-tree
and binary identities still remain those of their actual runs.
The existing FP32 report retains its original bytes and its original delivery
scope; its historical BF16 exclusion does not override the current precision policy and qualification.
Governance evidence does not extend the model's numerical or performance qualification.
Superseded cycle reports remain at their original commits; see [history](../history.md)
only for an explicitly requested historical investigation.
