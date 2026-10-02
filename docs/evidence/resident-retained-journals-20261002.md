# Valid-prefix retained device journals

Implementation `0fbc1b24ca8d8b178f83dfce32279ba23ce7e51f`, clean frozen
qualification; all eight bounded jobs passed.
[Audited source/build/result receipts](resident-retained-journals-20261002.json).

Aggressive training now copies only the valid prefixes of its own event, source,
emission and attention journals at a completed-window boundary. Conservative
policy keeps dense copies. Prefix row identities and aliases are preserved;
empty journals keep one unused row. Pending messages, outputs and KV retain
complete layouts. Dynamic nonzero extents synchronize at that boundary; no
reference trajectory or per-event host scheduling is introduced.

The next dense window is admitted before advancing, and actual retained bytes
are charged afterward. Consumer planning separately reserves packing metadata
workspace and still charges dense retained envelopes. This change does not by
itself relax original-wide training admission. Public structs and checkpoint
formats are unchanged; earlier C++ overloads keep their dense behavior.

Validation covers:

- 17 CPU capacity/head/interface checks, including C++/Python capacity parity.
- 44 NPU checks: seven compact-journal cases, three unchanged dense projection
  budget/lifetime cases and 34 actual consumer sample-slicing cases. Independent
  CPU autograd, three families, both schedules, aliases, None/zero roots,
  empty/different windows, optimizers and checkpoint suffixes are checked.
- Two standalone FP32/FP16 cells: 64 trajectories, 1,024 windows and 256 updates,
  independently compared with CPU FP32/FP64. Coverage includes cache and control
  profiles, SGD/AdamW, explicit owner maps and two-card to single-card resume.
- Separate installed-consumer two-card profiling: 46,612 Vector, 1,797 AI_CORE
  and 667 MIX_AIV rows, with zero observed AiCPU rows. It executes Attention,
  three physical sample groups and two complete AdamW updates; construction
  is included. This is finite trace evidence, not a guarantee for every operator.

No tests skipped or numerical tolerances loosened. The initial development
failure is retained: four fixtures requested the existing unsupported controlled
slot-affine VJP combination, and one omitted its explicit window seal. Corrected
fixtures use supported broadcast HST/SOFTP and specify the empty-window seal.

A separate same-source allocator comparison uses the common 128-body-node,
D128/B8/T4/V257 Attention packet, four B2 slices, one FP32 AdamW update and two
connected windows. Both runs use compact continuation storage and identical
physical chunk sizes. Only conservative/dense versus aggressive/compact journal
policy differs.

| Observation | Dense journals | Compact journals |
| --- | ---: | ---: |
| Peak NPU allocated bytes | 1,868,793,344 | 1,455,752,704 |
| Peak NPU allocated GiB | 1.74045 | 1.35578 |
| Maximum retained tape/state bytes | 388,563,192 | 184,628,448 |
| Saved continuation peak bytes | 2,755,300 | 2,755,300 |
| Loss | 5.612767696380615 | 5.612767696380615 |

Allocator peak falls 22.10%; retained tape/state bytes fall 52.48%. Both produce
3,145 events, 64 outputs and the same final cut; allocator peaks are within the
unchanged dense planning bounds. These are one-process memory observations on
separate leases, overlapping capacity/correctness work, with no throughput claim.
Live KV is still dense. Automatic sample admission, original-wide training,
CUDA and other CANN tuple qualification remain separate work.
