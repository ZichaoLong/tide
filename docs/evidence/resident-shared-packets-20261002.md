# Shared canonical communication packets

Implementation `a72868fa2506dc3ed60c5d248620afd09ac9cb68`.
The [audited record](resident-shared-packets-20261002.json) pins eight passed
fixed-source jobs, rebuilt host objects, unchanged core/CANN dependencies,
installed consumer, independent trajectories, allocator results and profiler
CSVs. Audit: `TASK/launchers/shared_packets_evidence.py`.

Canonical reduction and publication now preplan each phase's largest send and
receive packet on every device. All ordered groups reuse those arenas. The
existing peer notification protects sender reuse; receiver consumption and its
next copy remain ordered on the same stream. Send and receive roles have
separate storage. Contribution order, descriptors, connection flags, errors,
tails, payload precision and optimizer boundaries are unchanged. No kernel or
public ABI changes, host packet loop or reference-supplied numerical input are
introduced. Packet capacity still follows the original tensor budget and 64 MiB
endpoint maximum.

| Fixed-source validation | Result |
| --- | --- |
| Standalone/Python backends and installed client | Three builds passed; affected objects and unchanged dependencies checked |
| Existing transfer checks | 20 cases / 100 replays; two order-sensitive reduction replays |
| Shared cross-pair storage | 36 groups / 5 replays, opposite directions and local transfers; tails, None/zero, poison and error recovery |
| Canonical optimizer | FP32/FP16, 8 trajectories / 64 updates |
| Public sessions, ordinary and compact journals | 128 trajectories / 2,048 windows / 512 updates; independent CPU FP32/FP64 references |
| Explicit two-to-three-card restore | FP32/FP16, 32 trajectories / 512 windows / 128 updates |
| Both actual consumers | 33 checks, no skips; 32 recorded trajectories across graph families, schedules, precisions and automatic sample slicing |
| Separate actual FP16 Attention profile | Two cards, two complete updates; 53,274 operators, no observed AiCPU |

The shared transfer stress case reserves 4,346,630 bytes including metadata;
its total additional allocator peak across both devices is 4,720,640 bytes.
Communication storage no longer grows by a full packet for each of its 36 groups.

A same-lease comparison used one fresh process per version with identical
128-node Attention D512/B8/T4/V257, four physical B2 groups, FP32 AdamW and two
connected windows. All non-reservation counters, effective chunks and loss
were identical; loss was 7.532631874084473.

| Measurement | Previous packets | Shared packets |
| --- | ---: | ---: |
| Logical device 0 allocator peak | 8,940,336,640 B | 8,671,899,136 B |
| Logical device 1 allocator peak | 8,064,654,848 B | 7,796,217,344 B |
| Canonical reduction reservation across cards | 805,629,352 B | 268,758,440 B |
| Planned packet iterations over the step | 144 | 144 |

Each card's allocator peak fell by 268,437,504 bytes, approximately 256 MiB.
Reservation counters report the maximum across sample-group backwards, summed
across cards; they are distinct from measured allocator peaks. This is a storage
comparison, not a throughput recommendation or original-size training result.

Two failures remain recorded. Development component children all passed, but
CANN generated an untracked file in the source working directory and the source
guard rejected the parent. All frozen source hashes were unchanged; clean gates
use the output directory. The first D512 allocation baseline refused an undersized
module capability ceiling before execution. The corrected old/new pair used
the same physical 60 GiB/card admission, dimensions and safety margins with
larger hierarchical capability ceilings; neither failure was an OOM. The
successful profile's 50,726 AI_VECTOR_CORE, 1,825 AI_CORE and 723 MIX_AIV records
cover construction and execution; they do not establish an end-to-end speedup.
