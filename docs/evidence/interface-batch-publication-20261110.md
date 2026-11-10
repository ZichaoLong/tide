# Batched publication and descriptor capabilities — 2026-11-10

Implementation: `47704c9c44367ee2052cbf397a2b78cfa21314e7`; fixed mathematics:
`0770593`. [Machine-readable observations](interface-batch-publication-20261110.json)
retain exact build/configuration identities, original failures, costs and hashes.
The subsequent generalized Describe and purpose-specific recurrent queries are
not part of this source or these measurements.

The increment uses [read-only publication](../tensor-publication.md), whole-field
`unbind` at query/region boundaries, selective region metadata uploads and one
host transfer per unique returned value. Describe batching has a separate
capability from memory query batching. Compact windows now count **live** fibers
after consumption, preserving the explicit capacity failure instead of imposing
a cumulative-time limit. The [scheduling](scheduling-interface-audit-20261110.md),
[region](region-batching-audit-20261110.md) and
[recurrent](recurrent-interface-audit-20261110.md) reviews identify the affected
symbols; their overlapping scopes are not a whole-repository review.

## Correctness, training and continuation

| Fixed-source check | Result |
| --- | --- |
| Local query/sequence/region/publication/capability CPU | 501 passed, 13 retained numerical xfails |
| Graph/window/ports/returns/training CPU | 308 passed, 1 SDK-unconfigured skip, **4 strict failures** |
| Four failed affine cases, explicit diagnostic continuation | 4 xfails after all ten comparisons and exact resume |
| Same FP32 affine material promoted to FP64 | Python/native first evidence-root VJP passed |
| NPU-configured local flow | 122 passed, 3 retained xfails; includes CPU helpers |
| Literal owned-return NPU | 7 passed, 5 deselected |
| Installed Python and Python→native consumers | CPU/NPU independent small training and fresh-process exact restore passed |
| Standalone SDK | Build, 3 CTests and explicit owned-return CPU FP64/NPU FP32 checks passed |
| SDK complete/outputs consumers | CPU FP64/NPU FP32 three optimizer steps and fresh-process exact restore passed; complete mode checked independent roots |

The original `interface-batch` job remains failed. Its four failures are FP32
affine × DAG/PDG × Python/native, all at the first
`root.evidence[nodes.0.read_program.evidence_weight]` gradient, coordinate `(1,0)`:
absolute difference `1.3504177331924438e-6`, original atol `1e-6`, rtol `1e-5`.
The follow-up `interface-batch-rest` job passed its declared workflow, but its
four diagnostic cases remain xfails, not numerical passes. They completed the
remaining independent-root, optimizer, full-record and exact-resume comparisons
without another mismatch. The same FP32 parameters/inputs promoted to FP64 pass
the first-root diagnostic; scalar and candidates have FP32 distances from the
scalar FP64 result of about `1.6905e-6` and `3.0409e-6`. This supports a bounded
rounding diagnosis at this field, not a general tolerance relaxation.

Literal derivatives, independent schedules, partial/zero roots, shared parameters,
raw state, pending messages and fresh-process restoration establish different
properties. Small checks do not certify the full FC16 task's learning behavior,
all configurations or new scale/device combinations. No downstream installation
or dependency lock was upgraded.

## Finite costs

FC16: four serial regions × four nodes; adjacent regions fully connected;
B8/L128 = 1,024 input positions, d128, heads2, FFN512, summary32;
KV qry/Read+/cat/ind/context/cat/regional_total/quad, fractional k2.
Complete task 5,092,788 parameters, core about 4.564M; fixed downstream `0439efd`.
Python→native C++/LibTorch, FP32, auto window/packed-v2, minimal returns,
one native scheduler worker, CANN9.2/Torch2.12/TorchNPU2.12.

Each point used one process, one warmup/reset and one measured step/request.
Complete training includes preparation, scheduling/packing, loss, backward,
checks/clipping, optimizer and synchronization. Idle-only admission selected
physical NPU11; no project build overlapped formal timing. Other host workloads
were not controlled. These single observations are not a statistical speedup
claim, and the full cost task has independent numerical/learning `not_checked`.

| Placement | Host threads | Training s | Input positions/s | Inference TTFT s |
| --- | ---: | ---: | ---: | ---: |
| CPU | 1 | 12.446816 | 82.2700 | 3.158910 |
| CPU | 16 | 12.958704 | 79.0203 | not measured |
| NPU mixed-c | 2 | 17.212242 | 59.4925 | 5.098356 |

Training throughput is `1024 / complete_wall_seconds`, not generation throughput.
TTFT is measured from warm loaded local prompt creation/upload through first-token
host return, excluding tokenizer/queue/subsequent decode. Complete inference
request walls were 3.160353/5.106228 s (CPU/NPU), or 324.014/200.539 input positions/s.
CPU forward/backward were 4.94224/7.40531 s; NPU 8.44530/8.57546 s.

The [previous increment](interface-snapshot-bank-20261110.md) observed CPU1/NPU
training 14.6283/20.6338 s and TTFT 3.1947/4.3911 s. Training decreased in these
observations; **NPU inference increased**. A reduced autograd graph does not imply
faster no-grad inference. Actual work remained 16,384 candidates, 8,344/8,345 Full
computations and 25,060/25,064 messages; gradient norms 1362.2411/1359.8636.
Counts alone do not establish route or gradient equivalence. CPU RSS was
3386 MiB; NPU host RSS 2755 MiB and device peak allocated/reserved 1694.89/1822 MiB,
including warmup.

## What the graph diagnostic measures

Separate B2/L8 instrumented processes gave the same outer 615 nodes on CPU/NPU.
The registered union decreased from 13,477 to **11,317**, SelectBackward from
3,200 to **640**, and UnbindBackward increased from 3 to 403. CPU actual backward
executions decreased from 12,324 to **10,156**; NPU now executed 10,140.
Read/Full/region private groups contain 5,248/4,174/668 nodes, respectively.
Groups overlap and must not be summed or added to the Python outer graph.
Explicitly saved unique storage is 20,546,960 bytes, not allocator peak or all
ATen saved tensors. These observations establish the local transpose reduction;
they do not certify full B×L operator-count independence.

## Reproduction

Build/install the fixed source with [development](../development.md), using
separate Python-owned and standalone NPU libraries. The JSON records exact
drivers, environment, test names, build hashes and observations. Run the affected
publication/query/region, component-window, ports and return-packing tests with
`--dtype both`; preserve the four original strict FP32 failures. Explicit numeric
diagnostic mode `TIDE_DENSE_NUMERIC_REPORT=NEW_DIRECTORY` completes their remaining
comparisons while retaining xfail. The later tracked diagnostic
`scripts/diagnose_window_describe.py` accepts fixed fixture/build paths and does
not modify the installed implementation.

Installed consumers and SDK complete/outputs restoration use the commands in
[the prior report](interface-snapshot-bank-20261110.md#reproduction). Run
`weighted-return-check` explicitly; the client's three CTests do not include it.
Cost uses `benchmark_dense_query.py` with the same options as that report;
separate diagnostics add `--batch 2 --length 8 --profile stages --autograd-census`.
No historical raw run directory or precomputed reference trajectory is required.
