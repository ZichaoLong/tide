# General Describe, recurrent query scope and resident diagnosis — 2026-11-10

Implementation: `554f2f5a00ea1e1dbff8f13c48c09dbb4366efac` (including resident
diagnostics `ef779d5`); mathematics remains fixed `0770593`. The later
`ebfe075` changes the operator observer and documentation, not computation.
[Recorded results](interface-heads-qualification-20261110.json) retain source,
build, fixture, installed-consumer, job and original-failure identities.

## Implemented scope

The default weighted packed prefill now uses column implementations for all
installed finite Describe heads: cat/match, ind/attention, cat/grouped/fixed
evidence, content/content_stats key, fixed/context/trainable gain. Leading B/L
axes are supported inside the column functions; adapters split fields once at
publication. This is a finite set of local implementations, not qualification
of the Cartesian product. Mem and Describe capabilities remain independent;
explicit legacy, scalar reference, device/worker/dtype fallbacks and graph
family applicability retain their contracts. Auto policy is
`kv-packed-v2+describe-batch-v2`; consumers must validate an upgrade instead of
claiming exact continuation from the previous auto policy.

[Recurrent Read/Full](../weighted-recurrent-query.md) now materializes only
consumed H/Z/W fields and query features. The complete raw state, parameter
versions, projection layout and convolution/cache validation remain. SSD's
unused delta arithmetic is removed, while its full projection and delta_bias
shape checks remain. None versus connected zero, distinct Read/Full snapshots,
parameter aliases and all uses' contributions are preserved.

Resident changes add budget/node/phase/expression and service/workspace
diagnostics. They do not lower any budget, replace the projection layout or
change scheduling. [The storage contract](../resident-storage-accounting.md)
distinguishes construction evidence from numerical execution and throughput.

## Fixed-source checks

| Scope | Observed result |
| --- | --- |
| CPU FP64/FP32 directed suite, installed 554f2f5 | 490 passed, 14 retained numerical xfails |
| NPU-configured Describe/training/window suite | 80 passed, 2 retained xfails, **1 operator-observer failure**; includes CPU-only formula helpers |
| Installed Python and Python→native consumers | CPU/NPU small independent training and fresh-process exact restore passed |
| Standalone SDK CPU FP64 | Kernel/formula check, explicit owned-return check and 3 client CTests passed |
| SDK two new recipes × complete/outputs | CPU FP64 and NPU FP32 all eight workflows passed; three optimizer steps and fresh-process 2+1 restore |
| CANN service component | Nested service count, workspace and repeated-window checks passed |
| CANN weighted flow | FP32/FP16 each passed 21 small graph/read/schedule combinations and boundary checks; complete records, pending and raw-state continuation; no new resident training qualification |
| CANN construction witness | Expected old local-budget refusal reproduced and storage difference assertions passed |

The generalized Describe tests include independent scalar sums/encoders, explicit
leading B×L dimensions, all fields, partial/zero cotangents, frozen/shared
parameters and a prohibition against invoking scalar heads from the candidate.
The recurrent tests cover unused-state None gradients, zero-connected cache
entries, empty/missing taps, times above 2**54, negative time, full parameter
shape validation and original version guards. CPU small formula checks,
independent graph schedules and exact same-stack continuation answer different
questions; none substitutes for full-scale task qualification.

Two default-entry compositions check complete records, independent descriptor
roots, all parameter/input gradients, two SGD+momentum steps, pending at a retained
cut, checkpoint/new-owner continuation and the exact next update:
SSD/state with match/attention/grouped and shared encoders; KV/qry with
direct_attention/context gain. The standalone fixtures exercise corresponding
configurations with their own generated material, four FC nodes, two regions,
d2 and B2/P2; they are not the Python fixture's identical tensor material.
Complete SDK mode checks independent roots; outputs mode checks its actual
selected-return execution. No downstream lock or installation was changed.

The 14 CPU xfails retain earlier radial-evidence and affine first-evidence
rounding witnesses. The latter completes all other training/restore comparisons
before reporting xfail; [the original failures and same-material FP64 diagnosis](interface-batch-publication-20261110.md)
remain separate. Global tolerances are unchanged. NPU's two numerical xfails
also remain failures in the recorded comparison reports.

## Observer failure and lifecycle

The new projection-count test originally observed only `aten.mm`. At the actual
TorchNPU 2.12 dispatch boundary the matrix calls remain `aten.matmul`; its count
was zero even though value/gradient tests passed. A separate fixed-source
dispatch probe captured **22** two-dimensional matmul submissions, each with
all four rows in its right operand. `ebfe075` observes either operator and
asserts both operands are matrices; its CPU FP64/FP32 checks passed. The NPU assertion follow-up passed (1 case, 3.54 s), recorded separately with its
test-source identity and the unchanged 554f2f5 implementation.
This does not change the mathematical candidate or claim 22 hardware kernels.

`interface-heads` remains failed: the driver accidentally replaced part of
`LIBTORCH_ROOT` while expanding its path placeholder, after native/CPU checks
and fixture export. `interface-heads-rest` resumed from SDK construction, then
failed on the original NPU observer assertion. `interface-heads-finish` completed
the remaining NPU installed/SDK/resident checks. No completed qualification or
cost was rerun solely to make an outer job pass.

## Measured local budget boundary

The construction-only witness holds its input, metadata and 64 banks of 32×32
FP32 parameters before constructing one projection. Device results:

| Quantity | Bytes |
| --- | ---: |
| Local allowance | 32,768 |
| Already-held parameter bank | 262,144 |
| Local program storage growth | 271,152 |
| Genuinely new storage in the owner union | 8,868 |
| First local use of borrowed storage | 262,284 |

The old row guard refuses because local program growth includes borrowed storage,
even though genuinely new owner storage fits. Global union deduplication already
works; the mismatch is between that remaining allowance and a local first-use
charge. This is an actual small witness, **not a repaired guard** and not proof
that it explains every historical SSD/FC16 refusal. The original coarse KV query
guard is another boundary. A single-bank projection and a justified composed
allocation bound remain designs for the next authorized round.

## Reproduction and maintenance

Use [development](../development.md) for fixed wheel, native and independent SDK
builds. The JSON records exact test inventories and all eight SDK command lists.
The small recipes are
[SSD/grouped](../../tests/fixtures/describe-recurrent-grouped.json) and
[KV/context gain](../../tests/fixtures/describe-kv-context-gain.json).
Generate with `prepare_core5_study.py --recipe RECIPE --samples 2 --positions 2`,
export static binary material, then run `check_query_sdk.py` with both return
modes as documented in [the query contract](../weighted-dense-query.md).

Build `services weighted-budget-construction weighted-flow` with
`build_device_control.py`, then invoke `verify_device_control.py` on an admitted
idle NPU. The construction case is rejected by the profiler entry point, so
initialization operations cannot be mistaken for numerical execution evidence.
All long work used frozen read-only tracked source, background.slice, explicit
time/memory bounds and 320 build workers. Actual NPU admission selected physical
11; shared-card permission was not used. CUDA changes have source review only
in this increment, with no new build or hardware claim.

The navigation review corrected stale default/row-wise descriptions, separated
low-level replay defaults from weighted eager batched defaults, restored links
hidden by a malformed table, and linked both SDK fixtures/identity owners and
the small resident witness. A bounded follow-up checked 17 related documents
and 602 local targets, with none missing. This is a concrete navigation check,
not a declaration that every repository file/function has been reviewed.
