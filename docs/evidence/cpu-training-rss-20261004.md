# CPU training allocation allowance

The stricter CPU admission change on clean
`c6ef22474f658ecb12dd11310c710781023932ab` passed108 affected checks and the
original-width Attention calibration that previously exceeded its estimate.
[Audit and retained failures](cpu-training-rss-20261004.json).

The existing6.25% CPU allocation allowance now covers learned parameters,
masters,gradients and optimizer storage during training. Construction still
charges only learned/master storage. Device budgets,safety margins,accelerator
estimates and mathematical execution are unchanged.

| Original-width Attention B8 /physicalB4 | Estimated peak | Observed RSS growth |
| --- | ---: | ---: |
| Previous implementation,retained failure | 245,300,487,792bytes | 248,721,317,888bytes |
| Corrected clean implementation,first recheck | 254,061,046,480bytes | 246,219,792,384bytes |

The completed recheck uses all17,521,117,376 parameters,D2048/T12/V50304,
ATen16/worker1,two connected windows and one FP32 SGD update. It produced192
outputs,cut408 and finite loss in425.637614531s. Its RSS fits the new estimate.
This is measured calibration for the declared workload,not a universal bound.

The same diagnostic job then tried ATen1/workers16. That second process reached
its900s limit and was terminated without a completed result. Consequently
`wide-eager-cpu-policy-recheck01` remains **failed/exit1**,and no result or speedup
is inferred for the second policy. The accepted first case is recorded
separately;the earlier underestimated run also remains failed. All relevant
services are terminal with empty cgroups.

The clean108 checks cover CPU FP32/FP64 and FP16 consumer training/splitting,
Python/C++ capacity agreement and unchanged accelerator geometry. Overlapping
diagnostic/integration work prevents a formal timing recommendation here.
Audit:`python TASK/launchers/cpu_training_rss_evidence.py`,where
`TASK=/mi/data2T/zlong/tide-execution-flows`.
