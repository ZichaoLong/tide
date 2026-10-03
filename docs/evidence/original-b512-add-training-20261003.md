# Original Add B512 complete training

Reviewed 2026-10-03. **Passed one actual original-size complete update** on clean
`26176de888013fda5eccfe039fa504e87c2e7e95`; not an extrapolated B4 result.
[Machine-readable evidence](original-b512-add-training-20261003.json) pins source,
input, installed library/consumer, launcher, admission and result hashes.
[Prior qualification](consumer-phase-timing-20261003.md) covers the actual
consumer and independent CPU observables/gradients/update/continuation at smaller
sizes. This run does not supply a full-size CPU gradient or update oracle.

| Item | Actual execution |
| --- | --- |
| Model | Original Add D2048/B512/T12/V50304; 9,468,053,696 parameters |
| Flow | TimedDAG, standalone LibTorch, resident online prefill, FP32 SGD |
| Physical execution | Nine NPUs; B2 × 256, two connected windows, one update |
| Construction | 69.389676023 s, separate from complete update |
| Complete update | 2170.549862239 s |
| Sample work | 2168.898689171 s, including input, forward/loss, backward, accumulation and continuation |
| Optimizer | 1.651173068 s, including final finite checks and publication |
| Outputs / input tokens | 12,288 / 12,288 |
| Logical events / cut | 1,183,429 / 408 |
| Loss | 30.50836181640625 |
| Maximum allocator growth | 43,432,802,304 bytes; every card within its estimate and usable budget |
| Saved continuations | Every device within its admitted pool; logical B512 preserved |

The independently initialized candidate consumes the original hashed packet; no
CPU-generated event, route or gradient drives it. Sample slicing accumulates the
complete logical batch before the single update. No warmup or profiler ran.

Admission used the already-reviewed [B4 phase diagnostic](original-width-add-phase-diagnostic-20261003.md):
`(sample × 128 + optimizer) × 1.15 = 2814.0674665565 s <= 3000 s`.
The old whole-update scaling remains `3041.4433348544 s > 3000 s`, and its
historical refusal is unchanged. Actual complete time is 0.7713211883 of the
margin-inclusive phase forecast. This validates feasibility for this input and
configuration; it does not guarantee linear scaling on other input-dependent work.
All previous memory/context limits, 3000-second actual-step check and 1.15
forecast margin were retained. The bounded child limit was 3180 seconds.

`wide-add-b512-phase-admitted01` finished passed/exit0 at
2026-10-03T00:59:36.653989+00:00. Its service is inactive with an empty control
group and its nine-device lease is released. Physical IDs 1,2,3,4,5,6,7,9,11
were remapped to logical 0–8. Raw records are under
`TASK/runs/wide-add-b512-phase-admitted01/assessment`; source is
`TASK/sources/phase-timing-clean01`; `TASK=/mi/data2T/zlong/tide-execution-flows`.
Audit: `python TASK/launchers/wide_add_b512_evidence.py 26176de888013fda5eccfe039fa504e87c2e7e95`.

This is one cold, phase-instrumented feasibility execution. Limited Attention
development builds and directed correctness work used other cards during it;
no allocator comparison or profiling overlapped. Consequently this is not an
isolated throughput recommendation or a CPU/mixed/resident comparison. Attention
B512 training, eager mixed multi-card, full performance matrix/repeats/profiles
and final integration audit remain open under ROADMAP F1–F7.
