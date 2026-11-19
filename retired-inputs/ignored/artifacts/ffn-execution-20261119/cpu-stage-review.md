# CPU medium stage review

Tool `a5853660`; production tree `89f2707e71d989340c12de310797a4c7f4c99f64`. Both jobs passed; 970 frozen tracked files verified. Four diagnostic replays are exact against their own normal update for readout, private state, parameters, gradients and AdamW. This is in-process replay, not new-process checkpoint or CPU/NPU equivalence qualification.

CPU: 32 threads pinned 120–151 / NUMA 3, shared host. B16/L128/chunk32, FP32; two warmup updates then one normal witness. Ordinary three-repeat CPU timing on 240–271 / NUMA 6 remains separate. Tide has 129,025,464 parameters; Transformer32 d496/FFN2048/31×16 heads has 129,039,856 and is an auxiliary reference, not A–D.

Normal, stage and operator passes remain separate. CPU `synchronize()` is a no-op; NPU stage/operator observers force device completion. Their cross-device times are organization evidence, not controlled hardware speedup ratios. No diagnostic time is divided by the NUMA 6 normal dataset.

| Local CPU pass / seconds | Tide | Transformer32 |
|---|---:|---:|
| One normal complete update | 15.976779 | 3.639855 |
| Stage forward | 3.409753 | 1.554021 |
| Stage backward | 13.066094 | 2.090595 |
| Stage optimizer | 0.125199 | 0.137035 |
| Stage training root | 16.623955 | 3.795804 |
| Operator replay elapsed | 26.174497 | 7.760452 |
| Sum of operator intervals | 18.624890 | 5.728203 |

The next table uses the same **mutually exclusive forward categories** for CPU and NPU. Every exclusive scope row is assigned once. It includes hook/caller remainder and is not kernel time. Allocation metadata is observer work. Full/prepare categories exclude the separately listed FFN.

| Forward stage seconds | CPU Tide | CPU Transformer32 | NPU Tide | NPU Transformer32 |
|---|---:|---:|---:|---:|
| FFN (including normalization and row preparation) | 0.488093 | 0.870458 | 0.069587 | 0.060531 |
| allocation.bisect | 0.019586 | 0.000000 | 0.182795 | 0.000000 |
| allocation.control.remainder | 0.013295 | 0.000000 | 0.020060 | 0.000000 |
| allocation.save_face | 0.004993 | 0.000000 | 0.011198 | 0.000000 |
| allocation.sorted | 0.083845 | 0.000000 | 0.106340 | 0.000000 |
| allocation.validate_solve.remainder | 0.058869 | 0.000000 | 0.089821 | 0.000000 |
| description | 0.150759 | 0.000000 | 0.058027 | 0.000000 |
| input.projection | 0.004229 | 0.000000 | 0.000735 | 0.000000 |
| memory.K.full | 0.358127 | 0.000000 | 0.165586 | 0.000000 |
| memory.K.prepare | 0.304663 | 0.000000 | 0.133983 | 0.000000 |
| memory.S.full | 0.084798 | 0.000000 | 0.058393 | 0.000000 |
| memory.S.prepare | 0.280830 | 0.000000 | 0.128475 | 0.000000 |
| messages.gather | 0.319995 | 0.000000 | 0.137337 | 0.000000 |
| observer.metadata | 0.001678 | 0.000000 | 0.008194 | 0.000000 |
| readout | 0.004194 | 0.000000 | 0.000770 | 0.000000 |
| reception | 0.433907 | 0.000000 | 0.068651 | 0.000000 |
| regional.scoring | 0.428293 | 0.000000 | 0.127582 | 0.000000 |
| regions.group | 0.004358 | 0.000000 | 0.003506 | 0.000000 |
| state.join | 0.042483 | 0.000000 | 0.024494 | 0.000000 |
| state.split | 0.009072 | 0.000000 | 0.016446 | 0.000000 |
| transformer.attention | 0.000000 | 0.114893 | 0.000000 | 0.065275 |
| transformer.attention_norm | 0.000000 | 0.031713 | 0.000000 | 0.016464 |
| transformer.block_remainder | 0.000000 | 0.221741 | 0.000000 | 0.073635 |
| transformer.final_norm | 0.000000 | 0.001072 | 0.000000 | 0.000507 |
| transformer.layout.rope_masks | 0.000000 | 0.001174 | 0.000000 | 0.001421 |
| transformer.output | 0.000000 | 0.053622 | 0.000000 | 0.009213 |
| transformer.qkv | 0.000000 | 0.107834 | 0.000000 | 0.010914 |
| transformer.readout | 0.000000 | 0.000007 | 0.000000 | 0.000041 |
| transformer.rope | 0.000000 | 0.130031 | 0.000000 | 0.094877 |
| unattributed_scope_remainder | 0.313686 | 0.021476 | 0.107135 | 0.004721 |
| Total forward | 3.409753 | 1.554021 | 1.519113 | 0.337601 |

CPU Tide’s FFN forward scope is smaller than Transformer’s, but its forward also includes KV/SSD preparation, reception, messages, description, regional scoring and state organization. The overall training gap cannot be read from FFN parameter count or forward MACs alone.

CPU allocation: 48 sorted calls over 90,112 rows, 8 invalid rows, 4 fallback calls with 7 requested / 7,168 executed rows. NPU baseline: 48 sorted calls, 375 invalid rows, 28 fallback calls with 324 requested / 49,152 executed rows. Invalid includes inactive rows; it is not the fallback count. The fallback still executes 40 FP32 bisections plus 3 corrections over each full width 8 rectangle. CPU metadata adds 52 scalar reads, separately from 1,386 training-path reads and 1 outside loss-reporting read; these are not NPU synchronization counts.

| Reverse operator pass | CPU Tide seconds / calls | CPU Transformer32 seconds / calls |
|---|---:|---:|
| aten.bmm.default | 9.351598 / 2425 | 0.000000 / 0 |
| aten.mm.default | 0.001812 / 8 | 2.290412 / 1024 |
| aten.index_put.default | 1.934496 / 1660 | 0.000000 / 0 |
| aten.new_zeros.default | 0.369883 / 1836 | 0.000000 / 0 |
| aten.add.Tensor | 0.933101 / 5737 | 0.169054 / 2047 |
| aten.mul.Tensor | 0.913186 / 4449 | 0.200211 / 2821 |
| aten.slice_backward.default | 0.082242 / 867 | 0.099107 / 704 |
| aten._scaled_dot_product_flash_attention_for_cpu_backward.default | 0.000000 / 0 | 0.126532 / 128 |

Tide’s visible backward BMM sum is 9.351598 s over 2,425 calls. One prominent group is 20 calls of `[8,768,512] @ [8,512,192]`, with first-operand stride `[393216,1,768]`, totaling 1.152206 s. This is a transposed-stride batched contraction witness; no BLAS kernel or threading cause was measured. Generic backward rows lose forward module ownership, so these durations cannot all be assigned to FFN.

CPU Tide attention scopes expose BMM, `_safe_softmax` and `exp`, with no fused attention dispatcher observed. CPU Transformer has 128 `_scaled_dot_product_flash_attention_for_cpu` forwards and 128 corresponding backwards. NPU Tide has 28 native fused-attention forward/backward calls; NPU Transformer has 128. A decomposed dispatcher trace does not establish that the Python SDPA API was absent.

| Directly visible FFN forward multiply-add pairs | CPU | NPU |
|---|---:|---:|
| Tide | 184.704565 billion | 184.478073 billion |
| Transformer32 | 199.715979 billion | 199.715979 billion |

The count is `batch_product × M × K × N`, summed over directly visible matrix primitives. CPU Tide uses owner-batched BMM; CPU Transformer exposes flattened MM. NPU Transformer exposes `matmul` forward and an opaque `matmul_backward` dispatcher in reverse. Fused attention and opaque reverse internals are excluded from this arithmetic proxy; missing visible MM never means zero work.

Initial parameter and input hashes match between CPU/NPU for each subject, but warmed states and backend execution differ. CPU Tide KV selected/executed FFN rows are 118,518/335,616; NPU baseline has 118,515/335,104. Source identity and nominal shapes therefore do not establish the same activity trajectory or actual work.

CPU allocator records are empty. Process lifetime RSS includes construction, normal reference and warmed/diagnostic CPU snapshots; it is not an activation ledger and must not be compared directly with NPU allocated HBM. This review runs no models or devices. Raw identities, exact replay checks, shape/stride witnesses and full mutually exclusive cuts are in [cpu-stage-review.json](cpu-stage-review.json).
