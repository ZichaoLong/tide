# Immutable attention parameters across retained windows

Implementation `38858d02bd986e166ddaae1914b3ee1719bf34d0` is qualified by the
[audited record](resident-attention-snapshots-20261002.json). Nine fixed-source
jobs passed: standalone/Python backend builds, installed consumer build, four
correctness jobs, one allocator comparison and one separate profile.
Audit: `TASK/launchers/attention_snapshot_evidence.py`.

Each backward group now shares immutable copies of attention QKV/output matrices,
parameter biases, decay and pooling weights. Dynamic KV, history-dependent
log-bias, lengths and journals remain separate for each window. The cache binds
the actual forward parameter bank, including when grouped execution creates fresh
gathers, and validates identity, version, layout, geometry and aliases. Direct
CANN writes need not increment ATen versions: the existing prohibition on parameter
publication with outstanding tapes is essential. Backward, detach and close release
the cache. Public ABI, core library and device kernels are unchanged; all users of
the changed private training layout were rebuilt or reused with verified source,
object and compilation identities.

| Fixed-source check | Result |
| --- | --- |
| Retention/lifecycle Python tests | 16 passed, no skips; exact-byte admission and one-byte-short retry, fresh gathers, stale/replaced bank rejection, released-cache output lifetime, nonzero updates and disk restore |
| Actual Python-native and standalone consumers | 24 passed, no skips; three families, both schedules and precisions, complete updates against independent CPU |
| Native ordinary/compact retained training | 128 trajectories, 2,048 windows, 512 updates; FP32/FP16 and independent CPU FP32/FP64 |
| Explicit two-to-three-device restore | 32 trajectories, 512 windows, 128 updates, both precisions |
| Separate actual FP16 profile | Two complete AdamW updates on two cards; 53,274 operators, no observed AiCPU |

One fresh process per version used the same two-card lease, identical 128-node
Attention D512/B8/T4/V257, four physical B2 groups, FP32 AdamW and two connected
windows per update. The prior version already includes shared communication
packets. Loss remained exactly 7.532631874084473; all non-memory counters and
effective chunks matched.

| Measurement | Per-window copies | Shared snapshots |
| --- | ---: | ---: |
| Logical device 0 allocator peak | 8,671,899,136 B | 8,396,636,672 B |
| Logical device 1 allocator peak | 7,796,217,344 B | 7,520,954,880 B |
| Retained bytes, maximum backward group across cards | 1,998,020,368 B | 1,451,692,776 B |

Each card's allocator peak fell by **275,262,464 bytes (262.51 MiB)**. The retained
parameter saving was 546,327,592 bytes across cards; logical retained bytes and
allocator peaks have different accounting. The complete-consumer estimator was
not reduced. The profile includes construction and cleanup; its 50,726
AI_VECTOR_CORE, 1,825 AI_CORE and 723 MIX_AIV records support execution attribution,
not a throughput comparison. No original-size Attention training fit or speedup
claim follows from this calibration.
