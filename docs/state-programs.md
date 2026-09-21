# State program and first SSM profile

Implemented; qualification is in `evidence/m5a-state-programs.md`. `State` consists
of a read vector, named tensor slots, last logical observation time and observation
count. Every slot participates in trace, clear, in-memory detach and checkpoint v3.
`Next` still precedes Full and cannot read its value. The generic reset for these
profiles zeros read/memory tensors but preserves clock/count metadata.

The native `StateKernel` interface supplies initial, step, independent-sample
batch, optional exact sequence/packed-sequence blocks, Read, reset and validation. Clients can
provide their own immutable C++ implementation through `NodeWeights::kernel`;
the standalone `cpp/test/custom_kernel.cpp` is an example. No Python callback is
made by native node workers. Default batch and sequence methods are explicit
loops; exact-sequence capability must be declared before the planner uses it.
The step interface receives complete source-tagged fibers as well as content.

The first new memory profile is a diagonal input-selective SSM. For row-vector
content h, memory m, and learned parameters:

```
dt = softplus(h @ W_dt)
a  = exp(-softplus(A) * dt)
m' = a * m + dt * (h @ W_b)
read_vector = (h @ W_c) * m' + skip * h
descriptor  = sum(read_vector * read_weight)
```

This advances on observations; it does not decay during empty logical times.
The sequence implementation uses associative affine scan; the streaming batch
implementation stacks only actual samples for the same node. This is a declared
representative recurrence, not a Mamba/Mamba-2 checkpoint-compatibility claim.

EMA and SSM also implement packed sequence blocks: equal nonempty event lengths
form one `[time,batch,width]` affine scan, with separate initial memory per sample.
No padded event becomes a candidate and no recurrence crosses a sample boundary.
Python frontier uses this grouping; native `packed=False` keeps B=1 and
`packed=True` groups samples. `prefill=False` retains causal steps. Counters
distinguish semantic sequences from actual grouped scan calls. Final persistent
read/memory tensors are compact clones; intermediate trace views may share block
storage. This establishes batch/sequence execution, not a measured speedup.

The SwiGLU Full profile uses a 2*width intermediate dimension:

```
g = h + (silu(comparison @ W_gate) * (comparison @ W_up)) @ W_down
```

It then applies the existing HARD/HST/SOFTP Emit. Normalization/convolution,
model-specific layouts and arbitrary FFN program registration remain separate
extensions. The region selector is still the declared count/descriptor profile.
