# Current evidence

The five-workstream implementation has separate source-scoped records:

- [Measured hotspot choices](roadmap-hotspots-20261118.md).
- [Three Attention families and actual operator witnesses](roadmap-attention-20261118.md).
- [Sparse, shape and useful-combination comparisons](roadmap-supplementary-20261118.md).
- [Actual 0.5B/1B four-class scale comparisons](roadmap-scale-small-20261118.md).
- [9B/17B scale and actual capacity](roadmap-scale-large-20261118.md), including the explicitly pending CPU17B work at this stage.
- [Multi-NPU placement, transport failures and corrected launch environment](roadmap-placement-20261118.md).
- [Initial precision and rounded-allocation diagnosis](roadmap-numerics-initial-20261118.md),
  then [continued VJP, repeated backward and restored updates](roadmap-numerics-followup-20261118.md).
- [Bounded whole-model capture boundary](roadmap-resident-20261118.md).
- [Clean installed wheel and actual restored updates](roadmap-installed-20261118.md).

These reports preserve both failures and passes; the live handoff and outstanding
matrices remain in [STATUS](../STATUS.md). Each links its reviewed JSON evidence.

- [FP32 instance delivery, 2026-11-18](instances-fp32-20261118.md), with
  [complete reviewed observations](instances-fp32-20261118.json).
- [Instance governance and project cleanup, 2026-11-18](governance-cleanup-20261118.md),
  with [preservation, deletion and installed-smoke records](governance-cleanup-20261118.json).
- [Follow-up removal of released downstream inputs and small remnants](cleanup-followup-20261118.md),
  with [exact retirement and preservation records](cleanup-followup-20261118.json).

Each report identifies source, configuration, actual tests and remaining limits.
The existing FP32 report retains its original bytes and its original delivery
scope; its historical BF16 exclusion does not override the new target in ROADMAP.
Governance evidence does not extend the model's numerical or performance qualification.
Superseded cycle reports remain at their original commits; see [history](../history.md)
only for an explicitly requested historical investigation.
