# Complete resident consumer memory admission

Qualified source: **0b1a5aa68061f37aef1f4387cf28abb0c77a464f**.
[Machine-readable audit](consumer-capacity-20261002.json) records all eight
terminal passed jobs, source/binary/loader hashes, case results and retained
failures. Core and resident backends are byte-verified unchanged dependencies.
The independent CPU consumer is rebuilt; the NPU consumer reuses checked objects
with a fresh installed package, link and loader. This does not claim a new
from-scratch core/CANN build.

The complete resident Add/Attention consumer now plans per-device simultaneous
lifetimes before allocating model tensors. A positive `--device-memory-bytes`
caps incremental demand, additionally bounded by live driver free memory.
Conservative/aggressive headroom is25%/10% plus128MiB. Capacity-driven physical
halving changes neither logical input batch nor KV/journal capacities, loss
normalization or optimizer boundary. [Contract and limitations](../consumer-capacity.md).

Eight CPU checks passed, including24 C++/Python shape-plan comparisons, matching
actual declared parameter inventories, heterogeneous-card bottlenecks, immutable
state refusal, precision/retention costs, checked overflow, and existing CPU
consumer observations. Nineteen NPU checks passed with no skips:

- Eight complete continued training comparisons: native Python client and
  independent LibTorch, Add/Attention, FP32/FP16, streaming/prefill and SGD/AdamW.
  All force capacity-driven splitting and compare full observations, gradients
  and updates against independent CPU execution.
- A refusal before Python model construction; eight mixed/resident memory and
  CLI boundary regressions; two independent LibTorch inference trajectories,
  including delayed PDG Add and Settle Attention with FP16.

The larger calibration retains128 body nodes/544 edges, D128/B2/T2/V257,
17,384,240 parameters, FP32 Attention/AdamW and two cards. Warmup plus measurement
cover four continued windows and two complete updates. At3GiB incremental/card,
four physical reductions select Full/emission/reverse rows1, attention rows1,
key tile8 and head rows64. All logical capacities remain unchanged.

| Bytes per logical card | Card0 | Card1 |
| --- | ---: | ---: |
| Complete estimated peak | 2,756,542,808 | 2,114,316,504 |
| Observed construction peak | 294,161,920 | 250,509,824 |
| Observed warmup peak | 857,463,296 | 685,577,216 |
| Observed measured training peak | 854,292,480 | 682,924,544 |
| Prior wider-chunk training peak, including warmup | 940,328,448 | 845,570,048 |

The new loss is5.662106990814209, versus5.662106513977051 from the independent
earlier CPU and wider-chunk NPU calibrations (absolute difference4.77e-7).
Output count8 and final cut80 match. No reference events, routes, numerical
outputs or gradients enter the candidate. This is capacity calibration, not a
formal timing comparison; development/qualification work was concurrent.

Retained development failures: a later head-budget rejection exposed inconsistent
overflow diagnostics; the matched SDK's ACL header path and direct ACL link
needed correction; Python capacity adaptation initially assumed a diagnostics
field absent from its public limits. They were fixed without relaxing semantic
assertions or tolerances. A fast-test topology fixture also initially used invalid
fanout/local-span arguments and was corrected. No OOM search was used.

The planner is a conservative **estimate**, not an allocator quota or proof about
every vendor allocation. CPU/mixed total admission, full-size F6 throughput and
CUDA execution remain unqualified. Other shapes/stacks need peak calibration.
No device kernel changed; no new profiling or AiCPU claim is made. Prior device
profiles retain their original scope. Reproduce the audited evidence with:

```bash
python TASK/launchers/capacity_evidence.py 0b1a5aa68061f37aef1f4387cf28abb0c77a464f
```
