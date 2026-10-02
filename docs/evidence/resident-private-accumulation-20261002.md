# Private gradient accumulation values

Implementation **b3a6a24870b365905ffdfb24a16abedfa5373815** is qualified by all
[eight clean jobs](resident-private-accumulation-20261002.json). Audit:
`TASK/launchers/private_accumulation_evidence.py <full implementation SHA>`.
Standalone LibTorch and Python-native builds/clients remain separate.

The first accumulation still copies public backward exports. Later accumulations
consume only that private FP32 numeric bank; each 256-element device tile owns
its disjoint read/write range. New connection flags stay separate from the old
flags, preventing block zero from changing a flag while another tile reads it.
No CANN kernel, public ABI, core, numerical summation order, consumer estimate,
`max_bytes` admission or safety margin changed. All owners are preflighted and
built before execution; already-returned backward views remain stable.

The directed native gate passed **32 trajectories, 768 windows and 96 updates**,
including FP32/FP16, SGD/AdamW, connected-zero/None, retained cache gradients,
compact sample continuations, 2→legacy and 2→2 complete-cut restoration, and
independent CPU FP32/FP64 references. Each of four processes also repeated the
same50 boundary cases:28 numerical cases at widths1,63,255,256,257,8193,1048583;
21 budget/overlap/upstream-error checks; and one empty registry. These verify
separate flags, disconnected NaN poison, tail tiles and unchanged right inputs.
They are50 distinct cases, not200. Python passed14 accumulation/context tests,
including public exports retained across later accumulations. Actual complete
consumer comparisons passed24 cases, with no skips.

One old/new independent process pair used the same two-card lease and identical
D512/B8/T4/V257 Attention, four physical B2 groups, FP32 AdamW, two connected
windows and one update. Loss7.532631874084473, all work/retained counters and
admission records matched exactly. **Whole-process allocator peaks were unchanged**:
8,368,268,800 and7,520,954,880 bytes. Removing the transient replacement bank did
not lower the measured peak in this workload; backward still requires separate
optimization. No whole-graph throughput or original-size improvement follows.

A separate FP16 actual-consumer profile covered three physical groups, two
updates and both cards: **53,182 operators, zero observed AiCPU**. Profile timing
is not throughput timing. All jobs terminated and released their device leases.

Builds reused the two source/compile-command-matched development objects after
hash validation, with fresh links. Other host objects, core, all CANN kernels
and unchanged consumer objects were byte-verified and reused. This is not a
claim that every dependency was recompiled from scratch. Raw builds, snapshots,
JUnit, device logs and profiler CSVs remain under ignored task artifacts.
