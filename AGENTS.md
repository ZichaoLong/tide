# Working on Tide instances

## Re-entry
1. Run `git status --short --branch` and `python scripts/status.py`.
2. Read docs/STATUS.md and docs/ROADMAP.md. Do not resume retired P1–P6 plans.
3. Mathematics is the immutable snapshot in math/; identity is docs/upstream.json.
   Run `python scripts/check_upstream.py`. Do not follow mutable upstream HEAD.

## Current purpose and priorities
Semantic correctness first, performance second. Deliver the current FC and HB
main bindings. General graph foundations are a long-term possibility, not a
short-term requirement. Old teaching bindings and LH-DAG/PDG are excluded.
Execution and performance use FP32. FP64 may be used for accuracy verification;
FP16/BF16 implementation and validation are not part of this task.
The user authorized a complete rewrite without consulting or reusing the retired
implementation/tests. Git preserves that history. Preserve user untracked files.

Production code need not expose mathematical action records, dependency traces,
independent internal gradient roots or universal state decoding. Check model
values, selection, parameter gradients and necessary continuation directly.
Do not build generic frameworks or compatibility wrappers before useful models.

## Work
Use the relevant repository skill for development, optimization or maintenance.
User instructions override historical skill/doc requirements. Share code where
useful; separate instance/backend implementations are allowed. Tensor batches
should cross computation boundaries without per-event tensor reconstruction.
Three Attention families need separate shape/device/performance decisions.
Choose serial tools, concurrent calls or agents by expected benefit; coordinate
edits/resources, review and integrate one coherent result.

Only edit the authorized branch. Reference repositories and public software are
read-only. Do not send downstream messages without authorization. Keep data,
task heads, losses and model-quality research in downstream experiments.

## Verification and execution
Write new small formula tests from the fixed mathematics. Test all candidates
and useful combinations, not their Cartesian product. Keep meaningful gradient,
multiple-update and continuation checks. Finite rounding discrepancies get
bounded diagnosis; do not disguise support, causal or state errors as rounding.
Record actual dtype/operator/device, elapsed time, throughput denominator and
peak memory. Grad-enabled forward is not training. Count real candidates/Fulls
when explaining sparse performance. Use optimized external comparisons.

Long builds/tests/benchmarks use frozen source in detached background.slice
user services, bounded memory/time and durable logs. Available CPUs/memory and
idle NPUs may all be used; coordinate jobs instead of imposing a thread cap.
Shared NPU use follows current conversation authorization, never a new-session
default. Never stop other projects' jobs. Public toolchains remain unchanged.

STATUS is the current handoff and ROADMAP is the backlog. Keep brief configuration,
run/result and useful performance notes; no exhaustive traceability ledger.
Update handoffs atomically, commit tested increments, and keep implementation and
measured claims distinct. Push only under task authorization (granted here).
