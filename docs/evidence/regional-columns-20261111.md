# Regional tensor columns — first interface stage

Implementation: `2d2f8de` (full identity and commands in [machine record](regional-columns-20261111.json)).
Fixed mathematics: core-5 `0770593838a37c790a568226930567ed60f99b83`, allocation §2–6/S.2.
This is one implemented boundary of the authorized whole-chain plan.

`allocate_columns` and `region_columns` accept complete R×N tensors (R may flatten B×L),
retain real candidate domains, and return controls without an event-object round trip.
Python and C++ default regional VJPs consume private columns directly. Their analytical
allocation transpose carries structural row support independently of numerical zero bars;
legacy independent event roots remain in the outer adapter. Scalar references stay
independent; Python breakpoint/Newton and C++ bisection solvers remain distinct.
The [contract and symbols](../weighted-region-batch.md) link definitions, shapes and tests.

Fixed clean source qualification: 203 CPU FP32/FP64 checks passed (literal solution,
directional derivatives, branch/None/zero/version behavior, twelve scoring/sending
combinations, independent roots, publication and diagnostic observers); 27 CPU FP64
checks passed for general PDG/HB, retained windows, AdamW and checkpoint recovery.
Standalone LibTorch's kernel executable also passed its FP64 literal column solution/VJP.
Reused development binaries were checked against the committed C++ source hash and every
binary hash; the record preserves their original build provenance. This is not a fresh
NPU build or device qualification.

Development failures remain recorded. Two compiler errors were repaired; a metadata test
incorrectly required equal host-read counts and now checks that the wrapper adds none.
The original FP32 PDG/direct-attention task-gradient strict failures remain in
`test_weighted_region_execution` (case 4); no threshold or assertion was removed.
A finite diagnostic found the Python old/new region boundaries give the same discrepancy.
Promoting the same FP32 parameters/inputs to FP64 passed every gradient field: maximum
absolute difference 5.7e-14 for Python and 5.7e-13 for native. This supports a bounded
roundoff interpretation for this witness, not a general numerical safety claim.

One finite FC16 cost observation used B8/L128/d128/H2/FFN512, FP32 CPU/ATen1, Python
calling C++/LibTorch, auto prefill/query and minimal records. Complete training took
5.2125 seconds (1024/5.2125 = 196.45 input positions/s); core forward was 2.6893s and
backward 2.3271s. It executed 16,384 candidates, 8,344 Full and 25,060 internal messages.
The historical default measurement was 5.8351s, direct FC 1.3902s. This is an orientation
comparison: the older default used a Python-owned NPU binary on CPU tensors, whereas
this observation used a CPU build. It is not a new strictly equivalent paired speedup,
and the large remaining gap is not closed. No TTFT or new NPU throughput was measured.

Reproduce from the implementation commit with the project environment and a matching build:

```sh
python -m pytest -q tests/test_weighted_columns.py tests/test_weighted_region_batch.py \
  tests/test_region_batch_publication.py tests/test_weighted_allocation_observer.py \
  tests/test_allocation_failure.py --dtype both
python -m pytest -q tests/test_weighted_region_execution.py --dtype float64
BUILD/tidegraph-kernel-check --device cpu --dtype float64
```

Use the existing `benchmark_dense_query.py` entry with `--fc-shape fc16 --batch 8
--length 128 --device cpu --threads 1 --mode training --client native --baseline graph
--trace-records minimal --query-execution auto --phase-fences step`; pass the fixed
downstream repository and matching native build as documented in the machine record.
Long commands use a frozen checkout and a bounded background.slice service.
KV producer/consumer columns, message routing, typed continuation and the default whole
FC chain remain in ROADMAP; this regional stage does not certify those unfinished parts.
