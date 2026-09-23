# Frozen three-family CPU suite

Definitions: `benchmarks/foundation-v1.json`; exact topology/module/variants:
`scripts/foundation_workloads.py`. Twelve medium logical configurations and two
large presets only. This suite is a graph-only workload, with no vocabulary head,
embedding or LH parameter import. It complements the retained [LH/PDG source
kit](evidence/cpu-comparison-kit.md). That kit remains inference-only, with its
D16/B4/V257/6-step/warmup2 smoke; it has no new training-mode argument.

In a matching Torch Python environment, one command builds and runs graph smoke:

```bash
python scripts/benchmark_foundation.py --device cpu --tier smoke --build \
  --build-dir build/foundation --output-dir artifacts/foundation-smoke
python scripts/benchmark_foundation.py --device cpu --tier medium \
  --build-dir build/foundation --output-dir artifacts/foundation-medium
python scripts/benchmark_foundation.py --device cpu --tier large --describe
python scripts/benchmark_foundation.py --device cpu --tier large \
  --build-dir build/foundation --output-dir artifacts/foundation-large
python scripts/foundation_report.py artifacts/foundation-medium/suite.json \
  --output artifacts/foundation-medium/review.json
```

Every output directory must be new. Medium requires three independent process
repeats and two whole-workload warmups with all state/cache/ledger reset. Large
defaults to one bounded evaluation, not a speedup claim; each process includes
setup/warmup/measurement/profile within300s. Medium default timeout180s. Explicit
bounds may be chosen up to1800s before launch. Failures stay failed. `evaluated`
means assessment ended, not every workload passed. Stop scaling a large family
after its first failed stage; larger stages remain explicitly unlaunched.

`--ids`, `--families`, `--variants` and `--modes` select existing frozen cases;
unsupported combinations fail. Shapes are fixed in medium; smoke substitutes
D16/B4/T6 and training window3. TR01 exposes actual `grad-forward`, `backward`,
`optimizer`, `train-step` phases. Each window computes the declared mean roots,
performs backward/AdamW as required (epsilon1e-5, decoupled decay), and detaches
all continuation tensors. Full step includes loss/zero_grad/detach; individual
phases do not. Grad-forward is not training throughput. Optimizer timing includes
only the update; forward/backward prerequisites run outside that interval.

CPU timing uses synchronous client wall time and includes binding/record conversion
and scheduling. Model construction/input preparation are recorded but excluded.
Python reference variants retain semantic traces; native wall passes use trace=false.
This difference is explicit and prevents attributing their whole gap to a kernel.
All inputs are deterministic; seed7 matrices are normalized by width and scalar
transport weights are0.8, independent of edge index. Every sample position has
an explicit occurrence and application time. Ragged A01 uses four fixed length
fractions. Denominators use sum(input lengths), not ports; records include
ms/effective-sample-position, positions/s and ms/batch-position (wall/T).
Streaming ends at the declared finite cut, retaining pending messages; it does
not drain feedback graphs. Input-position throughput is not completed-output
throughput. For the frozen large PDG, stride1/T6 leaves three output positions
per sample and in-flight tail messages; DAG/Settle use sealed position clocks.
Check actual output/pending/work counts before comparing timings.

A separate accounting pass records actual projections/matrix FLOPs, logical
Agg/Upd/Read/Next/Full, source/event batching, selected/candidate events, padding,
cache size, replay counts and worker durations. `logical.Upd` counts nonidentity
body updates; other logical totals include encoded boundary events. The
`body_candidate_events`/`body_selected_events` counters permit body-only
comparisons. Measured timing is published before the separate profile starts;
profile failure keeps the whole run failed while retaining its completed phase. FMA counts as two operations;
elementwise/norm/softmax and general backward FLOPs are excluded. Forward FLOPs
in training include scalar semantic replay. Its worker-duration sum is not wall
time and is never added to wall phases. Backward remains ordinary autograd over
the scalar semantic oracle. Profile and wall durations are retained separately.

Resource discovery intersects affinity/cpuset and every ancestor CPU quota;
memory uses host total/current availability and ancestor limits/remaining space.
Half-effective budgets apply to the whole suite. Runs are sequential. Default
ATen1, native worker limit4, BLAS1 and interop1; Python and explicit serial
baseline variants use1, while worker/frontier/Settle variants use the limit; native constructor thread IDs, process
threads, Torch/OpenMP/BLAS pools, affinity and NUMA are recorded. No attention-head
worker pool is claimed. Build jobs default2. Real worker-group plus coordinator
RSS is sampled/enforced; cgroup usage is recorded separately and can include
unrelated account workloads. Timeout/cancellation terminate and reap child groups
and adopted grandchildren before another workload starts (up to10s TERM and20s
KILL/reap grace beyond the work deadline; failure to reap stops the suite).

Large presets use four active ranked regions of32 or64 candidates, budget1 per
region, plus dormant regions to reach the target parameter count. Each node has
independent same-fiber Attention/SwiGLU owners; exact counts precede allocation.
Stages use4×denominator,16×denominator and the target node count. The frozen
large PDG variant is native-stream-packed (serial, actual worker1); --workers
is a limit for worker variants, not a request to change that baseline. Large
DAG/Settle frontier variants use the requested limit. Parallel PDG has separate
medium and retained17.27B evidence; this fixed narrow assessment does not measure
its parallel ceiling. PDG/TimedDAG
use the legal positive-delay layered graph; Settle adds two boundary nodes.
Dormant Settle ranks increase its clock stride, changing KV decay intervals;
clocks, activation, KV and actual matrix work are reported. Equal parameter
scale does not establish function or active-work equality to LH's two cortex
namespaces. These bounded evaluations do not replace retained wide/narrow LH
comparisons or authorize another tuning search.

`run.json`, `metrics.jsonl`, `summary.json`, worker records and process audits are
local truth. Trackio is a best-effort local projection (`--tracking off|required`
is explicit); unavailable/degraded ingestion does not erase raw evidence.
Formal timing requires clean source, matching C++ content and every binary hash.

A clean source export works without the original checkout or Git metadata:

```bash
python scripts/export_foundation.py --output-dir /new/path/tide-foundation
cd /new/path/tide-foundation
python scripts/benchmark_foundation.py --device cpu --tier smoke --build \
  --build-dir build --output-dir artifacts/smoke
```

The export verifies every included source hash before build/run. Rebuild against
the target's Torch/LibTorch; binaries are not portable across architectures.
Required qualification is CPU FP32/FP64 on the recorded aarch64 host; other
hosts/backends remain extensions. Long commands use a detached project job in
background.slice/Nice10. No formal timing overlaps heavy task builds or gates.
