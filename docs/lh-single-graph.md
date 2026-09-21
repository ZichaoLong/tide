# Bounded IOCortex inference in a single PositiveDelayGraph

The adapter is implemented in the oracle layer; qualification is tracked in
STATUS until immutable-source evidence is available. It uses existing graph v13
clocks, source domains and slot-affine Full, without adding an execution primitive.
Original LH remains an inference reference, with its C++ source unchanged. This
construction covers the equal-width fixed-weight profiles of lh-iocortex.md.
It is not an arbitrary graph or pretrained-model importer.

## Construction and projection

For L body ticks per token, set D=L+1. Body tick kL+p maps to global kD+p;
readout token k maps to kD+L. Body StateClock is (D,0,L), readout is (D,L,1).
At global cut c the body cut is floor(c/D)*L+min(c%D,L), and the readout cut
is floor(c/D). The reserved phase adds no body decay step.

Each original body edge gets L phase-exclusive physical edges. Delays are one,
except the edge selected at phase L-1 has delay two. Body-output edges selected
at phase p have delay L-p and target the appended readout node. All outputs for
one token form one complete readout fiber. Vocabulary projection remains an
external rectangular head; graph payload width stays uniform.

Body aliases retain original incoming logical slots, receive/send scales and
local signaling weight/bias. Replicated physical output slots explicitly refer
to original parameter tensors; copying extra-parameter names is insufficient.
Body-output wires share the output scale and its identity projection, with
phase-specific receive scales from readout input ports. Readout logical slots
remain the L original phases. PortLayout stays bijective; all-softmax sees the
original logical source count.

The body projection maps every stored state timestamp, region history time,
input-ledger time, fiber/message send and arrival, physical edge ID and emitted
slot back to the body graph. It keeps input occurrence positions and tensor/cache
payloads. Messages on body-output wires reconstruct body outputs; those pending
are exactly the unfinished token-window buffer. All cuts, including before and
after readout, preserve these facts.

Readout projection compares local state/cache, history and labeled hidden/logits.
It omits the two-graph adapter's External.position ledger. LH has no such ledger;
a token number is not a phase occurrence count when phases/samples are missing.
The single graph has its own complete continuation. This projection is a readout
state/history view, not the complete two-graph readout continuation. No occurrence
counters or ledger are fabricated.

## Independent checks

- cpp/test/lh_single_graph.*: edge-major construction and projection from the
  original IOCortex fixture, with explicit parameter aliases.
- cpp/test/lh_single_compare.cpp: full body trace/state/history/fiber/slot checks
  with exact identities and the existing tensor tolerances.
- cpp/test/lh_single_check.cpp: native serial/parallel/packed streaming, every
  global cut versus the two-clock anchor, original candidates/logits and greedy
  IDs, whole replay versus incremental cuts.
- tests/single_graph_adapter.py: independent phase-major construction with
  physical IDs that differ from the C++ graph.
- tests/single_graph_checks.py: independent Python scheduling, complete-cut
  projections and direct checks against original exported values.
- tests/test_single_graph.py: cyclic/parallel-edge fixtures, nonunit scales,
  sparse samples/phases, zeros, six profiles, clear on/off, Python and native
  serial/parallel/packed/cursor.

Existing fixture-v1 files still describe the original two graphs; each language
reconstructs its single graph explicitly. New clocks/domains are not serialized
under the old schema. Manifests hash constructors and checkers along with the
original snapshot and exports.

Original think requires globally nonempty Pronounce windows. Manual ragged
think_single_step cases have no original Pronounce call. Empty readout windows
there only test Tide's no-candidate behavior, without claiming original-LH
readout parity. Whole-model training, composite optimizer ownership, unequal
widths and large-sparse speed remain separate obligations.

Norm comparisons follow the original whole-model rule: each descriptor matches
its own candidate's FP64 norm, and the cross-norm difference is bounded by the
actual candidate L2 perturbation plus FP64 roundoff. Candidate/control payloads
retain their dtype tolerances; discrete routes remain exact. The negative tests
reject both corrupted Read values and changed routes. No payload tolerance is
widened to accommodate this derived quantity.
