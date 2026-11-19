# Inference stage review

State: **completed_review**. Tool `46317a8d`, production tree `89f2707e71d989340c12de310797a4c7f4c99f64`.

Re-run: `python artifacts/ffn-execution-20261119/collect_inference_stages.py`. Optional `--cpu-transformer-run artifacts/<replacement>` and analogous flags select explicit replacement identities while retaining original v1 lifecycle records. This only reads frozen records and replaces the two review files atomically; it runs no models or devices.

| Case | Review / raw state | Supervisor | Completed diagnostic passes |
|---|---|---|---|
| cpu-baseline | completed_review / completed_finite | passed | stages, operators |
| cpu-transformer | completed_review / completed_finite | passed | stages, operators |
| npu-baseline | completed_review / completed_finite | passed | stages, operators |
| npu-transformer | completed_review / completed_finite | passed | stages, operators |

cpu-transformer selected `artifacts/ffn-infer-cpu-stages-transformer-v2`; original `artifacts/ffn-infer-cpu-stages-transformer-v1` remains `failed` (exit 2).

npu-transformer selected `artifacts/ffn-infer-npu-stages-transformer-v2`; original `artifacts/ffn-infer-npu-stages-transformer-v1` remains `terminated` (exit -15).

B8, prompt128/chunk32/decode16, FP32. Tide KV128 has 7.268409B parameters; fixed Transformer32 d4352/FFN15208/68×64 heads/window128 has 8.778371B. This is an auxiliary reference, not A–D or equal parameter count. Each job contributes one normal witness; formal repeated normal timing is separate.

Normal, stage and operator passes are independent requests. Prefill tables show the full 128-position total; decode tables average all 16 calls, including the first. Normal stable decode has 15 calls. Stage exclusive categories partition only that replay; release is separately recorded. No diagnostic percentage is computed against normal timing.

Stage exclusive partition: **prefill**, seconds for all 128 prefill positions.

| Mechanism | CPU Tide | CPU Transformer32 | NPU Tide | NPU Transformer32 |
|---|---:|---:|---:|---:|
| Attention_read_scope.memory.K.full | 0.206045 | 0.000000 | 0.033770 | 0.000000 |
| Attention_read_scope.regional.scoring | 0.411846 | 0.000000 | 0.079096 | 0.000000 |
| Attention_read_scope.transformer.attention | 0.000000 | 0.134621 | 0.000000 | 0.065514 |
| FFN_including_normalization_and_row_preparation | 6.933104 | 7.074455 | 0.893961 | 0.359847 |
| Full_query_packing | 0.097719 | 0.000000 | 0.108210 | 0.000000 |
| Full_query_publication | 0.108676 | 0.000000 | 0.029085 | 0.000000 |
| KV_Full_other | 0.699624 | 0.000000 | 0.122658 | 0.000000 |
| KV_history_field_gather | 0.094431 | 0.000000 | 0.117438 | 0.000000 |
| KV_history_order_including_compaction | 0.016202 | 0.000000 | 0.043034 | 0.000000 |
| KV_prepare_other | 0.938022 | 0.000000 | 0.102356 | 0.000000 |
| SSD_Full_other | 0.250643 | 0.000000 | 0.042131 | 0.000000 |
| allocation.bisect | 0.010613 | 0.000000 | 0.180331 | 0.000000 |
| allocation.control.remainder | 0.018356 | 0.000000 | 0.020242 | 0.000000 |
| allocation.save_face | 0.007531 | 0.000000 | 0.012236 | 0.000000 |
| allocation.sorted | 0.132100 | 0.000000 | 0.080246 | 0.000000 |
| allocation.validate_solve.remainder | 0.076482 | 0.000000 | 0.096614 | 0.000000 |
| description | 1.192226 | 0.000000 | 0.160633 | 0.000000 |
| input.projection | 0.004497 | 0.000000 | 0.001030 | 0.000000 |
| memory.S.prepare | 0.756485 | 0.000000 | 0.196176 | 0.000000 |
| messages.gather | 2.475702 | 0.000000 | 0.435941 | 0.000000 |
| observer.metadata | 0.001720 | 0.000000 | 0.008844 | 0.000000 |
| reception | 3.273493 | 0.000000 | 0.293475 | 0.000000 |
| regional_scoring_other | 1.722857 | 0.000000 | 0.077251 | 0.000000 |
| regions.group | 0.004508 | 0.000000 | 0.003486 | 0.000000 |
| state.join | 0.067734 | 0.000000 | 0.019578 | 0.000000 |
| state.split | 0.009530 | 0.000000 | 0.013850 | 0.000000 |
| transformer.attention_norm | 0.000000 | 0.080066 | 0.000000 | 0.015052 |
| transformer.block_remainder | 0.000000 | 0.535544 | 0.000000 | 0.068020 |
| transformer.final_norm | 0.000000 | 0.002646 | 0.000000 | 0.000458 |
| transformer.layout.rope_masks | 0.000000 | 0.002314 | 0.000000 | 0.001664 |
| transformer.output | 0.000000 | 0.691512 | 0.000000 | 0.026224 |
| transformer.qkv | 0.000000 | 2.030633 | 0.000000 | 0.058578 |
| transformer.rope | 0.000000 | 0.218958 | 0.000000 | 0.109330 |
| unattributed_scope_remainder | 1.118525 | 0.023132 | 0.262562 | 0.004846 |
| Total captured | 20.628669 | 10.793880 | 3.434233 | 0.709533 |

Stage exclusive partition: **decode**, milliseconds per decode call; all 16 captured calls including first.

| Mechanism | CPU Tide | CPU Transformer32 | NPU Tide | NPU Transformer32 |
|---|---:|---:|---:|---:|
| Attention_read_scope.memory.K.full | 30.814745 | 0.000000 | 7.192780 | 0.000000 |
| Attention_read_scope.regional.scoring | 21.459408 | 0.000000 | 12.086909 | 0.000000 |
| Attention_read_scope.transformer.attention | 0.000000 | 21.659153 | 0.000000 | 14.726858 |
| FFN_including_normalization_and_row_preparation | 602.883543 | 657.831517 | 169.699983 | 42.961359 |
| Full_query_packing | 7.271386 | 0.000000 | 15.152027 | 0.000000 |
| Full_query_publication | 6.809079 | 0.000000 | 3.814865 | 0.000000 |
| KV_Full_other | 90.582073 | 0.000000 | 26.121991 | 0.000000 |
| KV_history_field_gather | 32.285678 | 0.000000 | 36.475545 | 0.000000 |
| KV_history_order_including_compaction | 4.878416 | 0.000000 | 13.193085 | 0.000000 |
| KV_prepare_other | 159.679108 | 0.000000 | 21.737493 | 0.000000 |
| SSD_Full_other | 19.010133 | 0.000000 | 8.672237 | 0.000000 |
| allocation.bisect | 0.216642 | 0.000000 | 6.212824 | 0.000000 |
| allocation.control.remainder | 2.601624 | 0.000000 | 4.963073 | 0.000000 |
| allocation.save_face | 0.831089 | 0.000000 | 2.838230 | 0.000000 |
| allocation.sorted | 10.861089 | 0.000000 | 18.562057 | 0.000000 |
| allocation.validate_solve.remainder | 9.738764 | 0.000000 | 22.376424 | 0.000000 |
| description | 119.525772 | 0.000000 | 29.819289 | 0.000000 |
| input.projection | 1.221973 | 0.000000 | 0.227723 | 0.000000 |
| memory.S.prepare | 69.974367 | 0.000000 | 34.322355 | 0.000000 |
| messages.gather | 35.504757 | 0.000000 | 21.277032 | 0.000000 |
| observer.metadata | 0.348866 | 0.000000 | 1.467055 | 0.000000 |
| reception | 59.693515 | 0.000000 | 16.831507 | 0.000000 |
| regional_scoring_other | 128.107285 | 0.000000 | 14.596469 | 0.000000 |
| regions.group | 1.210970 | 0.000000 | 1.025685 | 0.000000 |
| state.join | 34.058461 | 0.000000 | 7.333144 | 0.000000 |
| state.split | 2.085215 | 0.000000 | 3.513905 | 0.000000 |
| transformer.attention_norm | 0.000000 | 9.123619 | 0.000000 | 3.134508 |
| transformer.block_remainder | 0.000000 | 57.652720 | 0.000000 | 17.150049 |
| transformer.final_norm | 0.000000 | 0.299985 | 0.000000 | 0.096723 |
| transformer.layout.rope_masks | 0.000000 | 0.285209 | 0.000000 | 0.332022 |
| transformer.output | 0.000000 | 60.316067 | 0.000000 | 3.900178 |
| transformer.qkv | 0.000000 | 186.542053 | 0.000000 | 9.369653 |
| transformer.rope | 0.000000 | 14.783617 | 0.000000 | 10.088643 |
| unattributed_scope_remainder | 78.657829 | 2.135111 | 22.773454 | 1.225149 |
| Total captured | 1530.311787 | 1010.629049 | 522.287140 | 102.985140 |

**cpu-baseline separate operator evidence:**

prefill: fallback 1 calls, 1 requested / 1024 executed rows; FFN float-bank gathers 156 calls, 95,806,291,968 logical output bytes. Counts here cover the whole phase, not its mean time.
Memory K: preparation projection rows 786,432; selected rows 77,528; executed FFN rows 467,336 across 32 calls.
Memory S: preparation projection rows 81,920; selected rows 81,920; executed FFN rows 81,920 across 20 calls.

decode: fallback 1 calls, 1 requested / 32 executed rows; FFN float-bank gathers 624 calls, 207,920,037,888 logical output bytes. Counts here cover the whole phase, not its mean time.
Memory K: preparation projection rows 98,304; selected rows 9,749; executed FFN rows 48,512 across 128 calls.
Memory S: preparation projection rows 10,240; selected rows 10,240; executed FFN rows 10,240 across 80 calls.

Diagnostic metadata identical: True; complete output SHA identical: True. Final int64 tensors without value hashes: {'stages': 20, 'operators': 20}. This does not compare normal output payloads or qualify checkpoint recovery.

**cpu-transformer separate operator evidence:**

prefill: fallback 0 calls, 0 requested / 0 executed rows; FFN float-bank gathers 0 calls, 0 logical output bytes. Counts here cover the whole phase, not its mean time.

decode: fallback 0 calls, 0 requested / 0 executed rows; FFN float-bank gathers 0 calls, 0 logical output bytes. Counts here cover the whole phase, not its mean time.

Diagnostic metadata identical: True; complete output SHA identical: True. Final int64 tensors without value hashes: {'stages': 0, 'operators': 0}. This does not compare normal output payloads or qualify checkpoint recovery.

**npu-baseline separate operator evidence:**

prefill: fallback 28 calls, 112 requested / 22528 executed rows; FFN float-bank gathers 156 calls, 95,806,291,968 logical output bytes. Counts here cover the whole phase, not its mean time.
Memory K: preparation projection rows 786,432; selected rows 77,528; executed FFN rows 467,336 across 32 calls.
Memory S: preparation projection rows 81,920; selected rows 81,920; executed FFN rows 81,920 across 20 calls.

decode: fallback 16 calls, 16 requested / 424 executed rows; FFN float-bank gathers 624 calls, 207,920,037,888 logical output bytes. Counts here cover the whole phase, not its mean time.
Memory K: preparation projection rows 98,304; selected rows 9,749; executed FFN rows 48,512 across 128 calls.
Memory S: preparation projection rows 10,240; selected rows 10,240; executed FFN rows 10,240 across 80 calls.

Diagnostic metadata identical: True; complete output SHA identical: True. Final int64 tensors without value hashes: {'stages': 20, 'operators': 20}. This does not compare normal output payloads or qualify checkpoint recovery.

**npu-transformer separate operator evidence:**

prefill: fallback 0 calls, 0 requested / 0 executed rows; FFN float-bank gathers 0 calls, 0 logical output bytes. Counts here cover the whole phase, not its mean time.

decode: fallback 0 calls, 0 requested / 0 executed rows; FFN float-bank gathers 0 calls, 0 logical output bytes. Counts here cover the whole phase, not its mean time.

Diagnostic metadata identical: True; complete output SHA identical: True. Final int64 tensors without value hashes: {'stages': 0, 'operators': 0}. This does not compare normal output payloads or qualify checkpoint recovery.

Actual output int64 times and named private `.times` values retain their recorded checks. SSD convolution `.conv.0` clocks have only metadata; floating KV/SSD/history values are not fully hashed. Matching diagnostic metadata/output SHA does not establish equality of all private state or clock values. Output hashing uses bounded 8 MiB host pieces outside capture.

CPU 32-thread affinity and NPU 4-thread physical-device identities remain in JSON. CPU synchronization calls are no-ops; NPU observers force completion. Diagnostic hardware ratios, best-batch performance and task quality are not inferred. Actual matrix/stride records, backend/dtype observations, KV-history/SSD preparation/state cuts, source/input/output fingerprints, numerical boundaries and pending/failed details are in [inference-stage-review.json](inference-stage-review.json).
