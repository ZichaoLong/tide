# Original IOCortexNet and two-clock graph composition

The oracle now calls actual unchanged `IOCortexNet::think` and
`think_single_step`, including embedding, four adjacency blocks, candidate state
updates, selection, activation/norm, clear and Pronounce. Development and clean
qualification are recorded in [evidence/lh-iocortex.md](evidence/lh-iocortex.md);
the qualified scope below is narrower than arbitrary LH configurations.
LH remains an inference reference.

## Mapping

Both cortexes form one body PositiveDelayGraph. Every original block reads
previous-tick activations, including input-to-output. All body wires have delay
one. For fixed inference weights, original next-tick signaling moves into the
current selected node's Full. Each original linear output slice has its own
slot-affine weight and bias.

| Block | Source offset | Target offset | Incoming order |
| --- | ---: | ---: | --- |
| inet | 0 | 0 | first, original intra CSC order |
| onet | N | N | first, original intra CSC order |
| iobridge | 0 | N | after intra, original bridge CSC order |
| oibridge | N | 0 | after intra, original bridge CSC order |

The token slot is last at inet node zero. Its declared all-softmax denominator
entry remains present on non-token ticks, when the actual message is absent.
PortLayout explicitly maps original CSR output slices and CSC source slots;
default external-first order would change attention and learned pooling.
Original physical edge IDs are retained, including parallel wires. The fixtures
deliberately reverse IDs relative to CSR positions to detect accidental mixing.

Each base hub has an independent count/affect region; other hubs use
full-capacity regions. Body state is observe-all, with optional selected clear
after taking the Full comparison snapshot. Counts stay separate for each cortex
and sample. Add and KV log-bias physical state decode through their own clocks.

Body output at onet node zero feeds the independently clocked singleton readout
through `token_inputs` (`token-window.md`). The vocabulary head is rectangular
and external to the uniform-width graph. Finish token k readout before supplying
token k+1 in generation. The oracle checks two greedy feedback tokens and replays
their recorded inputs as a whole teacher-forced window.

## Validation scope and navigation

- `cpp/test/lh_iocortex_fixture.*`: original module setup, graph/weight mapping.
- `cpp/test/lh_iocortex_compare.cpp`: cloned pre-selection proposals/cache,
  actual post-selection activations, all-node physical state, counts, emitted
  and pending messages with exact coordinates, complete-cut continuation.
- `cpp/test/lh_iocortex_oracle.cpp`: actual original entry points against native
  serial, three-worker unpacked and three-worker packed execution; readout also
  uses Frontier. Routes must agree exactly, without forced routing replay.
- `cpp/test/lh_iocortex_export.cpp`: versioned dtype/shape-explicit JSON containing
  original proposals, activations, physical caches, counters, pending and logits.
- `tests/iocortex_fixture.py`: independent Python time-major body and
  reference/frontier readout, whole versus cut execution against those values.
- `scripts/check_lh_selector.py --component iocortex`: snapshot/build/binary and
  fixture hashes. `check_lh_iocortex_python.py` rejects stale/incomplete fixtures.
  `develop_lh.py` archives development source; `qualify.py` runs the clean gate.

Small graphs have N=11 per cortex, width 4, two heads, batch 4 and vocabulary 7.
Six state/pooling profiles cross clear on/off, lead selection on/off, original
single/multi PACKED or CROSSBATCH (attention only), and three native schedules.
Each case uses one profile throughout both cortexes and readout.
The fixture policy couples clear with relu/L=3 versus silu/L=2, and lead with
bias+RMS versus no-bias+identity norm; it is not a full Cartesian norm/act sweep.
Original heap selection is used in PACKED modes, tensor selection in CROSSBATCH.
Four-token runs and ten ragged ticks cover missing samples, initial idle ticks,
present zeros, no token on intermediate phases, feedback and continuation.
Python exports use original single projection with lead+bias+RMS, both clear
settings, all six profiles and both token/ragged scenarios.

Each descriptor equals the FP64 norm of its own candidate. Cross-implementation
norm differences are bounded by the actual candidate L2 perturbation plus FP64
roundoff; FP32 candidates retain the existing FP32 tolerance. Selected routes,
history integers and message identities remain exact. Original active-softmax's
known FP64 multi-batch diagnostic defect remains an explicit unavailable case in
assertions-on builds; the separate consistent assertions-off build fills it.

This is bounded equal-width two-clock inference composition. Unequal widths,
arbitrary pretrained model import, whole-model training/optimizer checkpoint,
single-PDG containment and large-sparse speed remain separate obligations.
The single-clock construction is in `lh-iocortex-plan.md`; it is not yet proved.
