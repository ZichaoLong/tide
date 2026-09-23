# Independent native SettleGraph frontend

Implementation source: `ac19aad83dd97d7f7b567e22af189ba69277d088`. CPU aarch64, Python3.11.15 for
launching/tests, Torch/LibTorch2.10.0+cpu, GCC10.3.1, C++17/Release/C++11 ABI.

The C++ library now independently constructs a rank-aligned SettleGraph, encodes
boundary nodes/edges/local ports/source domains, maps parameter TensorImpl owners,
and runs a sealed [B,T,D] window through native frontier. Python performs no
encoding for this entry. `settle_projection.cpp` observes complete body cuts;
partial encoded continuation is retained, never reconstructed from token clocks.
Contract and build example: [Settle embedding](../settle-embedding.md).

Directed development source archive:
`artifacts/native-settle-dev-20260923-a/source.tar.gz`; tree SHA256
`c90ed396a142d43bdb15dfe57e172493b403b8cb0bf7d843ee32d4178153d77a`.
Build plus230 CPU FP64/FP32 tests passed in23.26s. Tests include all projected
trace tensors, slots, histories, external/internal source identity, parallel
edges and permuted ports, HST/HARD/SOFTP, node workers, packed/scalar execution,
input/initial-state/shared/unused parameter VJPs, independent roots, empty chunks,
streaming versus frontier, aliases and invalid ranks/cuts. EMA/SSM/event attention/
Linear/Gated Delta representative modules are covered. Native graph identity
matches an independently constructed Python encoding. This directed gate used
archived dirty implementation; the clean standalone qualification below tests
the committed source. Full six-stage acceptance is still open.

Clean native-only gate: `tide-native-settle-standalone-20260923-a`, detached
read-only worktree `/var/tmp/zlong-graph-execution-foundation/qualification/native-settle-ac19aad`.
Own output/build: `artifacts/native-settle-standalone-20260923-a/`.
CMake explicitly used `TIDE_PYTHON_BINDINGS=OFF`, then built only
`tidegraph-settle-check`, with2 workers. ATen/interop/OpenMP/BLAS1;
background.slice, Nice10, timeout1800s. Configure/build/FP64/FP32/ldd all exited0.
Standalone independent two-layer literal recurrence validates outputs, final
state, VJPs, unused Read=None, shared owners, actual length3 time prefill and
chunk/empty-window continuation. `ldd` has no unresolved library, libpython or
torch_python dependency. No Python headers are required by the core target.

Post-run audit verified456 tracked source hashes against Git archive,
binary and library hashes, every stage's actual exit code and both terminal
records. Unit inactive/dead, MainPID0, Result=success/ExecMainStatus0; no process
remains in its cgroup. Binary SHA256:
`4e3cc68a5173b2ae9adfe6f5dd06e29e1d6b8e5bd829843a95c02d111d1bfca2`.
Source inventories, exact commands, loader closure, timings and logs are in
`standalone.json`, `reviewed-audit.json`, `status.json`, `build.log`,
`float64.log`, `float32.log`, `dependencies.log`; launch and immutable driver are
`artifacts/native-settle-standalone-launch.json` and
`artifacts/check_native_settle_standalone.py`.

This verifies the finite rank-aligned broadcast-input/sum-output profile, not
arbitrary SettleGraphs or model imports. Packed training retains local semantic
replay. It makes no performance or new backend claim and adds no checkpoint
format. Initial embedding is restricted to cut0; complete body observation is
not an inverse encoding for adapter state. Named parameter handles survive but
boundary scale names map to explicit encoded agg_scale names.
