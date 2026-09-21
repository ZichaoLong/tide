# Local state-clock qualification

2026-09-22 (Asia/Shanghai), clean implementation
`bd933e0980a2076bace5c6ffbb09da61bc400593`.
**3745 tests passed in 300.89s**, including 265 clock-specific checks.
Both artifacts/clock-20260921-2031/status.json and verification/result.json
are passed, exit 0, empty dirty status at that source. Unit
tide-foundation-clock-20260921-2031 is inactive/dead, MainPID 0,
Result=success, ExecMainStatus=0; finished 2026-09-21T20:36:10Z.

Command: `python scripts/qualify.py --output-dir artifacts/clock-20260921-2031
--jobs 2`, through scripts/job.py in background.slice. CPU aarch64,
Torch/LibTorch 2.10.0+cpu, Python 3.11.15, C++11 ABI, two build workers,
OMP/OpenBLAS single-thread, backend autoload off. Source fingerprints and binary
hashes are retained with the build record. Graph identity is v13; checkpoint
payload remains v4. The gate reran all CPU tests and standalone native extension
checks, without rerunning the original LH oracles at their earlier source.

## Contract exercised

See [state-clocks.md](../state-clocks.md). Three immutable integers describe a
contiguous range of active phases per period. State kernels consume local ticks;
stored timestamps, messages, seals, Full/Read/Next and region histories retain
global time. Tensor/cache payloads and source tags are preserved. Invalid phases,
off-clock stored states, arithmetic overflow and callable Python policies fail.

Independent local-time schedules anchor EMA, SSM, Linear/Delta, event attention,
LH Add and learned same-fiber attention. Python/native serial, node-parallel,
packed and frontier paths compare outputs, complete state, history, input ledgers,
public-root VJPs and structural gradient absence. Cyclic phase-alias fixtures
combine SourceDomain with one/two-tick wires, compare every cut including reserved
phases, and check in-flight messages and physical idle decay. Both CPU dtypes
use the unchanged repository tolerances and exact discrete identities.

Optimizer/alias checkpoint round-trips, subsequent new inputs and their VJPs,
clock identity rejection, program sharing guards and explicit native decoder
clocks pass. Inference preserves underlying batching capabilities and does not
invoke training replay. Native custom kernel checks cover clock wrapping.

## Retained development evidence

artifacts/clock-dev-20260921-2020/ records a passed dirty-source build and 512
tests in 68.52s, with archived source and hash. Added continuation and pending-root
checks then passed (22 tests), followed by the final 265-test clock gate before
the clean commit. artifacts/clock-python-dev-20260922-a/ remains failed: the
wrapper initially assumed MatrixMemory exposed packed_sequence. The fix preserves
the existing per-segment sequence fallback and its work counts; it does not
claim joint batching or change numerical tolerances.

This qualifies the clock primitive, not the whole-model single-PDG LH mapping,
composite model ownership or large-sparse performance.
