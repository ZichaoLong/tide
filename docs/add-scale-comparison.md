# Fixed-graph Add scale comparison

The user requested a fresh local LH/PDG comparison after identifying the old
"8.8B" report as possibly Add. The existing232-node-per-cortex graph has
2208 logical edges; width2048/vocabulary50304 yields **9,468,020,899** Add
parameters. Dividing this count by1024³ gives8.818; conventional B means10⁹.
This is not evidence that the old run used exactly this graph or configuration.

The portable comparison entry now accepts `--memory attention|add` and
`--mode nograd|grad-forward`. Add is applied to both cortex namespaces and
Pronounce/readout, with all-softmax pooling, fixed computed retention0.99,
SiLU/RMS body Full, selected clear and the same two body ticks per token.
PDG uses its existing literal `lh-add-repeat-v1` kernel and `all_softmax`
Aggregate; source-domain phase aliases do not duplicate coefficient owners.
Retention is configuration in this fixture, not an additional learned owner.
Inference correspondence is limited to fixed weights. Training follows Tide,
including its packed semantic replay; LH training is not the authority.

`--lh-timer outer` removes ENABLE_RAIITIMER only from a newly prepared owned
copy. The main metric uses the surrounding steady-clock interval already
emitted as perf/token_seconds, without requiring Think log lines. The original
timer remains the default. `--phase-profile 0` disables PDG phase timers;
work counters and detailed profiling can also be disabled explicitly.
Both outer forward intervals exclude external token-ID generation and previous
logits disposal; embedding, body and vocabulary projection remain timed.
The scale check uses payload-precision tolerances for floating observables,
including FP64 norm descriptors derived from FP32 payloads. An initial FP32 Add
failure was a2.455e-9 absolute/6.20e-8 relative descriptor difference incorrectly
compared at FP64 tolerance. Payloads still use1e-6/1e-5, FP64 programs1e-10/1e-8;
dtype/shape/routes/identities/history and all state/message roots remain checked.
No execution formula or routing rule was changed to resolve this gate.
Grad-forward contains no backward, optimizer or detach and is labeled as such.
Same input token IDs and graph do not imply equal functions: parameters are
independently initialized. Native parameter counts and small independent PDG
scalar/packed/parallel complete-state checks precede large timing.

Bounded requested experiment: CPU FP32, D2048/B512/V50304, nominal local1/32,
56 physical CPUs from the permitted affinity, identical CPU set, ATen/BLAS1
for PDG node/head56, original LH ATen/OpenMP56 with requested OpenBLAS1.
Twelve token steps/warmup4; each of two engines and two forward modes gets
three independent process repeats, sequentially. Internal RAII/phase/operator
timers and work accounting are disabled for headline measurements. State is
retained throughout the token window; construction is excluded. No large
parameter file is written. Dynamic half-effective-memory and per-process time
bounds are recorded before execution; failures remain failures. Historical
3.3/3.6ms measurements and the earlier dirty-snapshot LH pilot are references,
not acceptance thresholds. No backward-performance claim is intended.

Completed and independently audited:4 native smoke cases and12 large runs, all
exit0. See [reviewed results](evidence/add-scale-comparison.md); STATUS owns the
current handoff. No further run is pending.
