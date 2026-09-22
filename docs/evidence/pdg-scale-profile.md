# Attribution of the comparable-scale PDG/LH timing gap

Instrumentation source:3151d5968b89479f1932f201e89b21af0e81a4aa.
This is a diagnosis of the [first scale result](pdg-scale-attention.md), not an
optimized executor result or an exact large-model LH equivalence claim.

## Records and correctness

Unit tide-pdg-profile-20260922-1145 passed/exit0, inactive/dead, MainPID0.
The clean read-only checkout is qualification/pdg-profile-20260922-1145 under
/var/tmp/zlong-graph-execution-foundation. Its isolated binary and build manifest
were copied from the development build; the exact C++ source and binary hashes
match. The manifest retains the build-time development revision.

Records: artifacts/pdg-profile-20260922-1145/; wide-profile/ contains the complete
run, native/root JSONL, stdout and summary. analyze.py verifies source, binary,
graph hash, terminal records, all8 token observations and interval accounting,
and writes analysis.json. Every work/model counter and logit checksum equals
the same token in the original PDG run. These are large-scale consistency checks;
full state/history/pending/route parity was tested only on small configurations.

Development gates passed62 tests/32.50s (pdg-profile-dev-20260922-1138), then
15 tests/14.48s for the input/head split (pdg-profile-head-dev-20260922-1143).
The scopes overlap; they are not77 distinct tests. Both retained archives and
terminal job records identify the tested source. CPU FP32/FP64, profiling
on/off and scalar/packed/parallel trace checks are covered.

The benchmark kept FP32,17,269,426,339 parameters, hidden2048, batch512,
workers160, ATen/inter-op1, seed7, packed row Emit, nograd and CPUs160–319.
Model construction226.98044s was excluded; whole-process peak105.02517GiB.
All tracked runs used best-effort Trackio, degraded because it is not installed;
portable local records validate. No dashboard is required.

## Same-window observations

The original37.33% difference uses tokens4–11: PDG33.75842 versus
LH24.58203ms/sample-token. The new profile has8 total steps, so its valid
comparison window is **4–7**:

| Run | Seconds per batch step | ms/sample-token |
| --- | ---: | ---: |
| Original unprofiled PDG,4–7 | 16.77750 | 32.76855 |
| New profiled PDG,4–7 | 17.40005 | 33.98447 |
| Original LH,4–7 | 12.13625 | 23.70361 |

The profile is3.71% slower than the old PDG prefix and43.37% slower than LH in
this shorter window. Shared-host variation and instrumentation have not been
isolated through repeated paired runs. The new phase values must not be attached
retrospectively to the old37.33% observation.

PDG coordinator wall times, summed over its three ticks per token:

| Disjoint phase | Seconds/batch step | Fraction of token time |
| --- | ---: | ---: |
| Aggregate/State/Read jobs and barrier | 7.01003 | 40.29% |
| Next/Full/row Emit jobs and barrier | 5.99091 | 34.43% |
| Region selection and comparison assignment | 1.27970 | 7.35% |
| Event/fiber grouping | 0.35387 | 2.03% |
| State/message commit | 0.40264 | 2.31% |
| Destruction of tick-local objects | 0.95880 | 5.51% |
| Output row assembly and vocabulary Linear | 1.39280 | 8.00% |
| Input preparation | 0.00209 | 0.01% |
| Advance overhead outside tick intervals | 0.00921 | 0.05% |

Tick intervals total15.99594s within advance16.00515s. They include job creation,
barriers, allocator effects and waiting, and are not per-operator CPU-time sums.

Original LH's disjoint nested timer leaves in the same4–7 window are:
Attention/chal6.349s; edge Linear5.19475s; norm/activation/clear0.17525s;
selection0.100s; gather0.14450s; other body work0.04175s; and embedding,
readout, vocabulary head and outer residual0.131s. Receive totals round to0ms.
Affected, Intra forward and Think single step contain other timers and are not
added again. The temporary PARALLELFORLOOP timer prints zero before its actual
interval; both prints are retained by the parser.

These are similar functional groups, not identical call boundaries or workloads:
PDG updates include Aggregate/Read and readout Attention, for example. Independent
weights and inputs can also change candidates and cache occupancy. LH's original
loop uses fresh unseeded `randint` token IDs each step; the earlier description
as greedy feedback was incorrect and has been corrected. PDG uses fixed IDs.

## Findings

**Serial management is material.** PDG spends2.99501s per batch step in region
selection, event grouping, commit and cleanup. The region phase alone is1.27970s
versus LH's approximately0.100s selector. `stream.cpp` schedules independent
regions on the coordinator; region_evaluate.cpp constructs and validates each
sample-region's result; lh_selector.cpp builds small control tensors and copies
history maps. State/message commit and temporary tensor/container destruction
also remain serial. Static CSR/CSC already exists; dynamic event/state storage
still uses many per-sample maps, vectors and tensor handles. The timer does not
separate each allocation, finite check or history copy within those phases.

**The dense head has a demonstrated runtime-configuration sensitivity.** Fresh
process head-only runs use the same FP32 input[512,2048], weight[50304,2048],
seed7, no grad, weight.requires_grad=true, two warmups and three measured calls:

| Startup configuration | Reported ATen / OpenBLAS threads | Mean Linear seconds |
| --- | ---: | ---: |
| PDG-like OMP1, MKL1, OPENBLAS1 | 1 / 1 | 1.366075 |
| LH-like OMP160, MKL160, OPENBLAS1 | 160 / 128 | 0.042366 |

All output bytes have identical SHA256 across these two processes. The isolated
operation is32.24x faster, saving1.32371s. This demonstrates that a substantial
part of the head cost can depend on runtime configuration; it is not a measured
end-to-end PDG improvement or proof that all graph kernels get the same benefit.
The probes use the same160-CPU affinity, but are small standalone workloads.

The loaded OpenBLAS is0.3.30, USE_OPENMP, MAX_THREADS=128. A fresh-process runtime
probe with LH's environment reports128 even though OPENBLAS_NUM_THREADS=1.
Thus earlier descriptions of **effective LH BLAS1** were too strong:1 was the
requested environment setting, not an observed runtime count. No live runtime
introspection exists for the original completed LH process; this is a replay
of its startup environment against the same library. Its inner at::parallel_for
context differs from its top-level dense head, so128 must not be multiplied by
160 and reported as the actual number of concurrently running worker threads.

Retained head-probe/ and head-blas-probe/ explored changing thread APIs after
initialization and showed little improvement. Those probes used
weight.requires_grad=false, unlike the matched fresh-process pair; they do not
isolate initialization order by themselves. head-operator-trace.json also shows
both F.linear and explicit matmul reaching aten::mm with equal outputs under
LH-like startup. A different mathematical operator is not needed to explain the
head discrepancy. All four tracked head records validate; head-analysis.json
retains their configurations, hashes and the negative observations.

**Node parallelism works.** The earlier matching PDG token1–3 serial/parallel
comparison is149.99953 versus21.78595ms/sample-token, a6.88515x speedup, with
matching work counters and checksums. The large profile has zero semantic
replays and uses row Emit; lack of parallelism, gradient replay and per-slot
matmuls are not explanations for this case.

**Packed-kernel details remain a separate attribution task.** Update and Full
consume most total time, but total share is not the same as excess over LH.
Current fiber Attention concatenates/stacks/clones per-sample caches and groups
by exact cache/query shape; a batch512 process averages about82 rows per update
call in4–7. LH uses its packed batched cache structures. These are optimization
candidates, but this profile does not assign a measured fraction to KV copies,
attention math, fragmentation, stragglers or route-dependent FLOPs.

## Next bounded changes

1. Record effective runtime pools and allocate threads separately for dense
   head/readout and node-parallel phases. Do not globally give each of160 node
   workers128 BLAS threads. Verify lifecycle and numerical behavior before a
   same-window end-to-end rerun.
2. Reduce per-sample selector/control/history overhead and use independent
   region work where safe, preserving hard/softp/HST contracts and validation.
3. Add compact reusable event/message/state batches to reduce serial commit and
   destruction costs; keep the simple executor as a numerical anchor.
4. Instrument Attention internals only when choosing its next optimization;
   match candidate/source/cache work as well as selected Full counts.

No optimized full-model rerun has been performed. The37% is an observed result
of this implementation and configuration, not an inherent PDG semantic cost.
