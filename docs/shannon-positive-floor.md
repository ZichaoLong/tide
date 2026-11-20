# Positive support for role-bounded Shannon allocation

Select `ModelConfig(allocation="shannon", shannon_support_policy="positive-floor-v1")` for serial or region-batched models. Direct callers select `allocate(..., kind="shannon", support_policy="positive-floor-v1")`. The training example accepts `--shannon-support-policy positive-floor-v1` with Shannon allocation. The model config and checkpoint record the selected policy. Missing policy fields resolve to `strict`, preserving its original serialized configuration.

With unit caps and a fixed total of one, scores `[-73.42066192626953, 32.303375244140625]` give a weak response of approximately `1.21516e-46`. FP32 round-to-nearest stores zero below half of `2^-149`. The selected policy stores this positive response as `2^-149`. With an FP32 cotangent of `1e30` on that output, the score derivatives are approximately `[+1.21516e-16, -1.21516e-16]`. Both senders are interior at the solution even though the strong output stores as one.

## Arithmetic and observation

The ordinary active-set solver works on the input device. A row with a free candidate at subnormal or zero storage and room above the role floors, or a row with an unresolved solve, is packed for host refinement. For each candidate bound the host solver evaluates the clamped total at that breakpoint against the target. The remaining free coordinates share the remaining mass in proportion to their centered exponentials. Decimal precision is 192 digits for FP32 and 1200 for CPU FP64. This path incurs synchronization, transfer and host arithmetic on an NPU.

A mathematically positive value that rounds to zero receives the least positive subnormal of the public dtype. Fixed zeros, true endpoints and representable subnormals retain their storage semantics. Each stored value must respect its role bounds. If `r` coordinates receive `delta`, the additional mass `r * delta` must fit `64 * width * eps * max(stored_sum - r * delta, low_total)`. The solver checks the row bounds and the constrained residual. An excessive perturbation raises `ArithmeticError`.

The first-order VJP uses the refined free faces and binary mantissa/exponent pairs of the pre-floor slopes. Role and total adjoints follow the allocator's existing tie conventions. Reception, Attention and KV consume the stored masses. Two unequal tails that both store as `delta` consequently contribute equally to a normalized read when their logits are equal.

```python
from tide import observe_shannon_support

with observe_shannon_support() as counts:
    output = model(inputs)
print(counts.refined_rows, counts.repaired_rows, counts.repaired_coordinates)
```

Counts include solves executed within the observing context, including recomputation. The policy's throughput and multi-device execution have not been measured. [Allocator](../src/tide/allocation.py) · [Refinement](../src/tide/shannon_support.py)

## Qualification on 2026-11-20

The qualified source consists of 26 Python files under `src/tide`. The SHA256 of the compact sorted JSON mapping of relative names to file SHA256 values is `cbf9c33157bd744e240fc4e662862cd147a5ba78861452838c053e8945d1e4f4`. Implementation commit `885f7ef96be8cc31814e509c687017e6ff922d72` contains exactly these files, including the cap-face repair from `049ecdcb`.

The CPU suite passed 322 allocation, model, storage, gradient and checkpoint tests plus 35 example CLI tests. The dedicated 16-test file passed on CPU and under the Ascend launcher, with 15 tests exercising real NPU FP32 and one using the CPU FP64 boundary. Coverage includes both sides of underflow, fixed/variable totals, real bound ties, broadcast roots, mixed ordinary/refined rows, multiple stored tails in reception/Attention/KV, and serial and region-batched models. [Regression tests](../tests/test_shannon_positive_floor.py)

The downstream experiment ran the original r09 failure batch and two actual successor batches with this source on CPU FP32 and Ascend FP32. Both repaired 23, 56 and 111 coordinates respectively. The restored third update was bitwise equal on CPU. On NPU the restored private state was exact. Maximum absolute differences were `1.87e-9` or less in parameters, `9.32e-10` in AdamW, `3.73e-9` in clipped gradients and `4.77e-7` in loss. The official strict checkpoint was also migrated for one separate NPU update. These are isolated mechanism and continuation checks. [Downstream acceptance and source lock](../../tide-learning-lab/docs/requests/TL-REQ-007.md)
