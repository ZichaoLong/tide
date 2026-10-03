# Original-width Attention fixed-owner diagnostic

Clean implementation `29effaed064d59b9da930ec3acec810be58b2e1a` completed one
independent original-width Attention update on 2026-10-03. The B512 cost gate
refused execution. [Audited record](original-width-attention-owner-diagnostic-20261003.json).

D2048/T12/V50304,480 body nodes,17,521,117,376 parameters; B4 split into four
physical B1 groups on eleven NPUs. Two connected windows,FP32 SGD,one complete
forward/loss/backward/accumulation/update,8 CPU threads,no warmup or profiling.
The explicit Full/state owner map and operator maxima match the static B512
plan. Original queue896,arrivals896,outputs64,trace3072,KV256,KV journal8192,
4GiB saved-context pool and60GiB/card budget remain unchanged.

Construction took282.182195511s. Sample work47.781389243s plus optimizer
2.636987134s gave50.418376377s. Loss20.07830810546875,outputs96,events9256,
stages224,final cut408. All allocator and saved-context gates passed; per-card
observed peaks ranged42,838,530,048–43,996,483,584bytes,below unchanged estimates.
The actual maxima and phase records remain in the JSON report.

The old whole-update extrapolation is7421.5850026944s; the phase-aware forecast
is7036.4530317737s,including the unchanged1.15 coefficient. Both exceed3000s.
The pilot job passed/exit0,its service is inactive with an empty control group,
and all eleven device leases were released. **B512 was not executed.** This
retained refusal does not close original Attention training or the performance
matrix. It is one cold feasibility diagnostic,not formal throughput or a
full-size CPU gradient/update comparison.

Raw records: `TASK/runs/wide-attention-owner-pilot01/assessment`; immutable
source `TASK/sources/owner-map-clean01`,build
`TASK/builds/owner-map-consumer-clean01`. Source,binary,loader,packets,plans,
phase arithmetic,terminal status and artifact hashes were audited using
`python TASK/launchers/wide_attention_owner_evidence.py
29effaed064d59b9da930ec3acec810be58b2e1a`.

Follow-up investigates useful physical batches under retained safety margins.
Any finite-horizon capacity change needs a static bound independent of actual
routes and a separate execution; a measured peak alone does not justify it.
