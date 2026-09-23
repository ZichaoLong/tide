# Optional native operator timing

The scale runner and portable CPU comparison kit accept `--operator-profile 0|1`
(default 0). LH requires a newly prepared copy with `operator_profile: true` in
its build manifest; older source kits reject the option. Coverage is CPU packed
row-emission inference. Unsupported grad/unpacked/slot runs are rejected.

`detail/*_worker_seconds` sums **exclusive calling-thread elapsed durations**.
Nested scopes are subtracted; these values include descheduling, allocation and
library calls on that thread. They are neither process CPU time nor coordinator
wall latency. Do not add them to `profile/*` intervals, divide by worker count to
infer latency, or use percentage shares as a causal attribution of a speed gap.
`*_calls` counts timed scopes/phases; `*_max_seconds` is the largest exclusive
scope. Reset and collection require all measured workers idle. One executor is
measured per process. Disabled scopes avoid clock reads and counter registration.

| Category | PDG | Original LH CROSSBATCH |
| --- | --- | --- |
| input_pack | ordered source scaling/stacking, or packed-source reuse/gather | local index preparation and CSR SumCoe |
| qkv | QKV Linear and query scaling/reshape | same |
| kv_build | functional cache append, decay, bias/mask and grouping metadata | indexed append/capacity growth, hidden list bookkeeping, decay |
| kv_gather | stacked KV; single policy also query-owner gather | BatchCachedConcat indexed cache gather |
| attention | scores, masking, softmax, weighted values | same plus attention argument/list preparation and custom Function dispatch |
| pooling | per-event source coefficient selection and weighted pooling | node-batched confluence |
| output | pooled-row stack and output Linear | output Linear |
| state_commit/state_other | state slicing, cloning and assembly | residual update argument preparation |
| aggregate/read/next/full_other | respective PDG local evaluators, children excluded | no corresponding instrumented scope |
| norm | activation and layer norm | activation, layer norm **and clear** |
| emit | row Linear | row Linear |

Some ownership work falls between these scopes. Exact packing's temporary vectors
are destroyed inside its state phase; single's final locals are destroyed after
its last phase. KV state clones in single are part of kv_build. These boundaries
must accompany comparisons. PDG `detail/state_commit` is local state assembly,
distinct from coordinator `profile/commit_seconds` that publishes states/messages.

With [packed transport](packed-transport.md), source stack/scale moves into
Aggregate and input_pack measures reuse plus any source-slot reordering. Batch
Next counts one local batch scope, including selected reset, instead of one
scope per event. Logical `work/next_steps` is unchanged; `work/next_batches` and
`work/next_reset_batches` report physical batches. Compare the combined scope
costs as well as the individual timers when work moves between phases.

Work counters (`op/*`) remain independent. Compare profiling on/off numerical
values and work inventories, then measure overhead on the workload being studied.
Timers perturb small calls; unprofiled paired timings determine performance.
Full-state checks and independent semantic anchors establish correctness;
large output checksums alone do not. LH changes apply only to owned prepared
copies, with source/binary hashes, and retain its computation and schedule.
