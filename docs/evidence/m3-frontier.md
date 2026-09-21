# TimedDAG frontier qualification, 2026-09-21

Clean tested source: `06d9d3eb5bd39f0a74c7f3122431f4965d03e759`.
Environment and compiler match `m1-streaming.md` (aarch64 CPU, Torch 2.10.0+cpu).
Command: `python scripts/qualify.py --output-dir artifacts/m3-20260921-0815`.
Build and qualification exited 0; **191 tests passed**, pytest 14.28 seconds.
Retained lifecycle/build log: `artifacts/m3-20260921-0815/{status.json,task.log}`;
test manifest and log: `artifacts/m3-20260921-0815/verification/`.

Adds Python and native frontier comparisons with the time-major oracle in both
dtypes and all three Emit modes, four profiles with region quotient cycles,
clear and selected-only adoption; forward traces and full-root VJPs; chunk
continuation; one/three native node workers; scan enabled/disabled comparison.
On a two-sample, eight-position, three-node chain, the frontier makes 3 stages,
3 batch/sequence Full calls, and either 6 state sequence blocks or 48 state
steps. These counters demonstrate actual batching, not a speedup measurement.

The precise scheduling and potential-domain limits are in
`../frontier-contract.md`. Generic operator plug-ins, SettleGraph encoding,
specializations, advanced memory and LH equivalence remain unqualified.
