# First new-instance CPU observation

Implementation: `e73bc03eca91bc5a8bb03209bb6802429039411b`.
Fixed math: `4f5b9f20a4030d68ee66ae2d26e34659a5e37234`.
[Machine-readable observations](instances-first-cpu-20261117.json).

Frozen-source CPU qualification passed 227 tests in 25.26s. This includes
independent FC and Timed HB chronological formulas/gradients, 21 parameter
inventory cases, SSD and Attention formulas, optimizer and checkpoint checks.
This source predates the later zero-reception KV continuation-gradient fix;
the long-window test that found it is a subsequent independent check.

FC is four regions/layers × eight nodes, d128, K, source_residual, encoded
regional attention, role quad with k1, independent parameters. B8/L128 means
eight independent samples, each containing128 input positions. Timing starts
from zero history. All values are FP32. CPU used32 ATen threads. The requested
AllowedCPUs0–31 and MemoryMax64GB were not enforced by this host's undelegated
per-service controllers; effective affinity was not recorded for this first
workflow. Observed process peak RSS stayed near2.1GB. One warmup and three measured windows per mode;
the table uses the median. Other host jobs were present.

| Subject | Parameters incl boundary | Inference positions/s | Grad-forward positions/s | Complete train positions/s |
|---|---:|---:|---:|---:|
| Tide FC4×8 |9,334,304|8,168|5,648|685|
| Transformer36, nearest parameter count at d128 |9,446,528|7,078|2,313|314|
| Transformer4, same graph layer count and d128 |1,049,728|66,415|16,380|20,311|

These are different functions and different work. The Transformer has batched
QKV, native SDPA, RoPE,4d SwiGLU and RMSNorm; it has no Tide preparation,
regional scoring/allocation or private node state. Tide's fresh inference
evaluated32,768 candidate updates,9,769 Fulls and66,312 positive messages.
k/n=12.5% is a strength-budget ratio; measured Full support was29.8%.

The throughput denominator is B×L=1024. Tide inference window latency was
125.37ms. This does not measure first-token latency or stateful decode.
Training includes zero_grad, forward/state packing/return, scalar loss,
backward, foreach AdamW and synchronization. Grad-forward never backpropagates.
RSS is cumulative per process; it is not an isolated model peak.

Some grad-forward/training samples varied substantially (all three observations
are retained). This first finite workflow is evidence of runnable models and
an initial comparison, not a stable speedup or scaling claim. Attention family,
region solver and Full packing costs require subsequent device measurements.

The NPU workflow at this source failed admission after its180s bound: every
eligible shared card had an existing cooperative exclusive lock. No NPU
operator was executed, so this is not an NPU numerical or performance failure.

Reproduction from this source with a matching Torch environment:

```
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest -q
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m benchmarks.run \
  --config configs/fc-k.json --device cpu --threads 32 \
  --subjects tide transformer_parameters transformer_layers --output NEW.json
```

The recorded actual argv is in the JSON. Long executions follow docs/execution-flows.md and use a
frozen checkout in background.slice.
