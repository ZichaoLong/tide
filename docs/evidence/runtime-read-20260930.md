# Runtime ownership and state-dependent Read — 2026-09-30

Clean runtime source `8244f13` and Read source `06db0c2` passed the gates recorded
in the [manifest](runtime-read-20260930.json). All nine jobs terminated with exit0.
The runtime build passed six CPU CTests; component builds passed four CPU CTests
and standalone loader checks. Eight fresh NPU processes exercised runtime lifetime.

Raw CANN owners now prevent the last runtime session from finalizing too early.
Five injected recoverable API failures check partial construction, finish/submit/
boundary-wait errors and retriable cleanup. Failed programs reject resubmission,
drain before releasing resources and permit ordered cleanup retries. A separate
finite-program quarantine worker retains owners, refuses finalization/new work,
prints its verification marker and exits86; the driver requires both marker and
exit status. This does not certify physical device hangs or driver-reset recovery.

The clean Read build passed all15 component cells. Its integrated forward profile
compares640 windows against independent CPU Streaming/Greedy:3252 events,2012
actual emissions,66 windows with multi-time node batches, including38 state-Read
batches. Content, old, proposal and mixed Read modes cover feedback, parallel
edges, unequal delays, absent/present-zero inputs, exact large clocks/counts,
active-only adoption, selected clear and continuation under another schedule.
FP32 tensors use rtol1e-5/atol1e-6; discrete coordinates, selection and histories
are exact. State-causal regions never batch more than one time per node.

The device prepares an ordered scratch state sequence when the declared module
contract permits it. Otherwise only that region/sample is restricted to its
first complete frame; other regions keep legal prefixes. The actual selection
and state commit determine the next iteration, without a CPU numerical route
prepass or fixture-specific schedule. See [the profile](../content-flow.md).

A separate content trace records101630 AIV and716 AI Core tasks, including914
state-Read kernels, with no AiCPU task or host-fallback diagnostic. It contains
644 model submissions and644 model-boundary waits for640 cases plus four
refusal/recovery executions. It also contains30236 ordinary stream synchronization
API calls from setup, boundary tensor operations and diagnostic/reference work.
One model submit per window therefore does not imply one host synchronization
across the entire diagnostic executable. This trace is placement evidence, not
throughput or a CPU/NPU speed ratio.

Integrated numerical execution remains FP32 inference with the explicitly supported
sum/identity-or-EMA/linear-Read/count-or-positive/adopt/identity-or-tanh/broadcast
profile. General modules, attention/KV, vectorized computation, FP16 integration,
public matrix, peer progression and complete training remain separate obligations.
