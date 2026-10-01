# Device parameter updates and forward publication

`DeviceOptimizer` is an internal single-NPU SGD/AdamW owner component with FP32
masters/gradients/slots and FP32 or FP16 payload owners.
It consumes [parameter-owner adjoints](resident-parameter-vjp.md), using the
existing named optimizer's canonical group resolution and option validation.
It is not a public training session, retained-window autograd or a full-size
throughput qualification. Unsupported module/dtype profiles continue to refuse.

Values, momentum/moments, optional AMSGrad maxima, update counters and Adam bias
corrections live in device buffers. Construction copies initial CPU parameters;
subsequent updates use device gradients and connection bits. Shared TensorImpl
aliases receive one update. Distinct TensorImpl owners sharing storage are
explicitly refused: independent packed values cannot yet reproduce sequential
mutations of overlapping storage. Gradient reduction itself still distinguishes
those owners.

None skips parameters, slots, counters and weight decay. Connected zero performs
the configured update, including momentum and decay. SGD preserves first-use
momentum initialization, dampening, Nesterov and maximize; AdamW supports betas,
epsilon, decoupled decay, maximize and AMSGrad. Options are stored as FP32 scalar
coefficients. Adam bias correction uses the stable recurrence
`c_next = (1-beta) + beta*c`, with the complement formed from the declared host
double, avoiding cancellation in `1 - rounded_beta_power`. FP32 rounding is
validated against independent CPU FP32/FP64 updates; bitwise identity is not promised.

Each step has device metadata preflight, packed numerical proposals, one finite
gate and a commit phase. Nonfinite participating values/gradients/slots/proposals
refuse before any live parameter or slot changes. Disconnected poison is skipped.
Counters remain int64; negative or exhausted counters refuse. Existing error
codes remain sticky. Finite and counter errors are20 and21. A runtime failure
still requires the enclosing training owner to become unusable; the component
does not promise recovery from a partially executed runtime submission.

Tensor admission includes live and proposed parameter/slot buffers, counters,
bias corrections, static tables and per-tile status. Per-tile scalar status has
separate cache-line storage. Empty registries have bounded dummy arguments and
produce no update. Construction rejects invalid groups, dtype, shape, identity
or budget before recording updates.

FP16 owners initialize FP32 masters from their quantized payload values. Each
participating proposal also checks the actual round-to-nearest FP16 conversion
before any live parameter, slot or counter commit. A finite FP32 value can still
fail this gate; error20 reports the refusal with all live owners unchanged.
The master itself remains unrounded, retaining updates smaller than one payload
ULP. In particular,65512 can remain a valid master because it rounds to finite
65504;65520 rounds to infinity and refuses. Checkpoint restoration validates
the same representability condition before writing any device state. There is
no automatic retry, loss-scale adjustment or skipped-update accounting.

`append_parameter_publish` records a packed copy from unique owners to all used
forward banks: Full, state, Read, input/Aggregate and delivery scales. Static alias
tables are allowed; parameter values and the error decision remain on device.
HARD Read can share a differentiable parameter and must then be refreshed despite
having no local gradient. Frozen CPU parameter copies do not drive later windows.
Forward diagnostics snapshot the source scales on device at window entry, so an
optimizer update cannot change the recorded provenance of earlier messages.

FP16 publication rounds masters directly into packed payload banks on device.
Normalized Aggregate and fiber-pool banks use FP32 storage of the quantized
payload value, so publication rounds to FP16 and then widens. It must not copy
the unrounded master into these FP32 banks. Event QKV/output parameters and all
used aliases share this rule. Publication consumes already validated masters and
a sticky update error; it makes no live writes after a refused update. The
internal bank view is available independently of graph VJP capability, allowing
this component to be tested before enabling complete FP16 training.

The internal training-step checker composes actual forward journals, graph VJP,
owner reduction, SGD/AdamW, finite checks and publication, then advances the same
device state into subsequent windows. Its CPU reference independently executes
the same inputs and its own updates. Window boundaries explicitly truncate
gradients in this checker. They do not establish a retained-window lifecycle.
Public ownership/checkpoint integration, retained graphs, other module adjoints,
complete FP16 training, peer training and the performance matrix remain separate
required work. `master-publication` uses shared synthetic owner-gradient packets
and independent CPU master updates plus continued inference. It verifies
publication and forward behavior, not graph-backward correctness or training
throughput; qualification status remains in STATUS.
