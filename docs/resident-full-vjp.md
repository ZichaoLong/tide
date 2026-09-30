# Device identity/tanh Full VJP

`tools/device_online/full_vjp.h` is an internal first-order FP32 component for
resident training. `ContentFlow::full_tape()` borrows the actual device event
journal and frozen identity/tanh parameter banks; diagnostics are required.
The owner rejects unsupported Full kinds before exposing this tape. Closed or
failed owners cannot expose it. The next advance overwrites the journal;
retaining a multi-window training graph requires separate storage and lifecycle.

For identity, only the content cotangent is connected. For tanh Full,
`g = content + tanh(comparison @ W + b)`, the component recomputes the activation
from the saved comparison and parameters. It never obtains the activation by
subtracting content from g. Weights/biases have a caller-owned zero sentinel row;
only real connected tanh owners and this sentinel enter batched matrix operators.

Values and connection bits are separate. None remains absent, connected zero
stays connected, inactive Full cannot receive a connected cotangent, and poisoned
absent values are never evaluated then multiplied by zero. One device metadata
writer validates the whole logical tape before numerical work and writes all
connection flags. Malformed count/owner/time/activity refuses with error 2;
unavailable kinds refuse with error 12. The caller must inspect the sticky error
at its boundary. No partial numerical result is exposed after preflight refusal.

Device control selects connected rows, advances bounded chunks and reduces
repeated per-node owners in stable row order. Feature tiles have unique writers;
padding gets distinct scratch destinations, avoiding conflicting index-copy.
The parameter outputs are per-node partials; sharing/alias reduction is still a
separate registry operation. An identity-only profile has no weight/bias outputs.

Tensor admission includes gradient outputs, sanitized banks and bounded matrix
work before allocation. The incoming tape/cotangents and the enclosing
`CannProgram` operator workspace have separate budgets. Empty and shorter tape
replay clears old numerical results, connections and chunk counters. Physical
chunk size changes neither the logical VJP nor the optimizer boundary.

The gate uses independent CPU FP32/FP64 autograd, repeated owners/samples,
nonaligned widths, large residual content, poisoned unused parameters and rows,
None/zero cotangents, chunk sizes 1/5, short/empty replay, explicit refusals and
actual resident forward tapes in streaming/greedy modes. Profiling is separate.
This does not implement graph message backward, HST/SOFTP, LH/SwiGLU Full,
public autograd, optimizer, FP16 or peer reverse progression.
