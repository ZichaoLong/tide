# Post-attention Confluence profiles

Keep `lh-fiber-attention-sum-repeat-v1` unchanged. Four additional names in
Node.memory select different output-row pooling; they share the same Q/K/V,
visibility, clock, cache and clear contract in `fiber-attention.md` and the
packed path in `fiber-packing.md`. No new graph field or implicit reinterpretation
is introduced. Profile names and parameter shapes belong to graph/checkpoint
validation; current schemas are owned by `semantics.md`.

Let J be the present local source slots and y_j the attention output for that
source's query, after joining its heads but before the output projection:

| Name between `lh-fiber-attention-` and `-repeat-v1` | Pooled vector | Original LH Confluence |
| --- | --- | --- |
| `sum` | sum of y_j | add |
| `mean` | sum of y_j / number of present sources | average |
| `linear` | sum of w_j y_j | linear |
| `active-softmax` | sum of softmax(w restricted to J)_j y_j | actsoftmax |
| `all-softmax` | sum of softmax(w over all local slots)_j y_j | allsoftmax |

Learned profiles use one `extra["fiber_pool"]` vector of length equal to the
graph-owned incoming slot domain, initialized to ones. Linear weights may be
negative or zero. Logits/weights must be finite and match payload dtype/device.
Mean and sum have no pooling parameter. Nodes with no incoming sources have an
empty domain and never generate an update; a real candidate always has rows.

Coefficients act **after attention**, before `fiber_out` and its once-per-event
bias. They never rescale or remove Q/K/V input rows. A query with a zero linear
coefficient still adds K/V and can change other query outputs. A numerical zero
source is present, including for mean/active-softmax denominators. All-source
softmax also includes absent slots in its denominator.

The independent Aggregate remains raw scaled-source sum for Content/Read/Full
and Tide's SOFTP/HST formula. It does not pre-apply post-attention coefficients.
Physical input/edge scales still multiply arriving rows before Q/K/V. Ordering
is by local slot, whose mapping must preserve original CSC plus bridge/token
ordering when importing LH. QKV/output matrices and absent original bias mapping
remain unchanged. No pooling coefficient is recovered by dividing input values.

## Training, sharing and state

Training follows Tide, not LH's custom backward. A used vector parameter has a
full tensor gradient: absent coordinates under linear/active-only pooling are
ordinary connected zeros, whereas all-source normalization can give them nonzero
gradients. This is a vector parameter contract; it does not pretend each coordinate
is an independently missing Parameter. Standard optimizers may apply weight decay
to all coordinates of a used vector. A cache-only root in an isolated node has
no path to pooling weights; downstream graph feedback may create legitimate paths.

Kernel factories receive incoming degree, independently of the shared matrices.
Native model validation calls `StateKernel.validate_policy`; the fiber program
checks the graph's profile, heads and domain even for an already supplied kernel.
Sharing an entire learned-pooling state program across different degrees is
rejected; projection parameters can still be shared between separate compatible
programs. Same-degree sharing, all memory slots, clocks and the coefficient vector
round-trip through checkpoints. In-memory cuts/detach/parameter-epoch rules are
unchanged. Packed schedulers retain semantic replay for first-order public roots.
Mean/active-softmax may coincide in value initially but remain distinct profiles
with distinct parameter/VJP contracts and graph identities.

## Validation boundaries

Analytic tests use two same-fiber inputs 1 and 3, explicit coefficients/masses
2/3/5 over three slots and a missing source. They check source derivatives,
coefficient-vector derivatives, slot permutation, post-projection bias, unchanged
K/V and the difference between missing and zero. Schedules cover scalar/packed,
parallel, cycles, specializations, embedding, continuation, optimizer and sharing.

The original attention oracle adds 72 weighted/mean configurations to the 264
sum cases: width=4, heads=2, biases enabled, decay=.01, clear on/off, PACKED with
single/multi projection and CROSSBATCH, each against Tide serial/packed/frontier.
Original active-softmax multi-batch uses full-domain softmax followed by active
renormalization; extreme logits can underflow there. Tide uses stable active-domain
softmax. Only finite, bounded oracle cases establish numerical agreement; no claim
is made to reproduce original overflow/underflow bugs or its training behavior.
Consult immutable evidence for which qualification has actually passed.

Original assertions-on FP64 active-softmax multi-batch is unavailable in this
snapshot: its diagnostic `SumCoe(num, indptr)` explicitly defaults to FP32 while
`w` is FP64. The default oracle checks this exact exception and reports the 12
affected configurations as unavailable; it does not count them as numerical
passes. A separate assertions-off build from the same immutable source checks
all 336 configurations per dtype, recorded separately as `oracle-release/`.
The `ENABLE_RUNTIME_ASSERTION` flag remains enabled for the ordinary oracle;
plain C/C++ assertions remain enabled in both builds. Snapshot hashes are unchanged. The first observed failure and source snapshot are retained at
`artifacts/fiber-pool-dev-20260921-1709/`.
