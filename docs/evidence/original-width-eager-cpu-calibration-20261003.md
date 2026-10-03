# Original-parameter CPU calibration: retained Attention refusal

Clean implementation `2c04005255b3d0b674cef302894bb9a5d0f0378e`.
`wide-eager-cpu-pilot01` ended failed/exit1 with an empty cgroup. The
[machine audit](original-width-eager-cpu-calibration-20261003.json) retains both
fresh-process results and verifies source/build/packet/terminal identities.

Both consumers use original480-node topology, D2048/T12/V50304 and complete
learned parameters, with B4/physicalB2, two connected windows and one FP32 SGD
update. LibTorch CPU, TimedDAG/prefill,16 ATen threads/one node worker, packed
sources/batch-next;512GiB host cap,80GiB learned-parameter cap,4GiB head cap.
Each child had900s and the finite pilot1850s, stopping at the first failure.

| Memory | Parameters | Complete update | RSS growth / estimate | Acceptance |
| --- | ---: | ---: | ---: | --- |
| Add | 9,468,053,696 | 59.403475322s | 104,698,310,656 /122,250,520,896bytes | Passed |
| Attention | 17,521,117,376 | 256.699170879s | 228,749,291,520 /227,592,210,112bytes | Failed memory calibration |

Both returned96 outputs,cut408 and finite losses (31.5859947205 /20.0782928467).
Attention executed forward/loss/backward/SGD but exceeded the model estimate by
1,157,081,408bytes, about0.51%. It is retained as failed; completing arithmetic
does not override failed memory acceptance. The wrapper preserved the full
measurements and returned nonzero. No OOM or memory-budget relaxation occurred.

Construction RSS also exceeds the individual construction-phase estimates at
both scales (the existing gate checks the maximum of all estimated phases).
This points to missing host allocation/retained-buffer overhead in the current
shape estimate. A corrected model must cover these observations and pass a new
fresh-process calibration before original B512 CPU execution. The original
refusal and all margins/budgets remain unchanged.

Source `TASK/sources/eager-capacity-clean01`, build
`TASK/builds/eager-capacity-cpu-clean01`; raw results under
`TASK/runs/wide-eager-cpu-pilot01/assessment/{add,attention}/consumer/result.json`.
Re-audit with `python TASK/launchers/wide_eager_cpu_evidence.py 2c04005255b3d0b674cef302894bb9a5d0f0378e`.
These are feasibility/RSS measurements with other qualification work present,
not isolated formal throughput, original B512 completion or full-size gradient
parity. CPU independent reference remains available and the protected historical
CPU task was neither resumed nor stopped.
