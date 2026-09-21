# Matrix memory profiles

Candidate qualification is tracked in STATUS. Both profiles advance on accepted
observations, preserve state during empty fibers and expose every memory tensor
as a named slot. Clear zeros every slot with the declared connected-zero VJP.
The current profiles use one memory head with key/value width equal to model
width. They are representative equations, not checkpoint-import adapters.

## Linear Attention

With `phi(x)=elu(x)+1`, `q=phi(h Wq)`, `k=phi(h Wk)`, `v=h Wv`:

```
M' = M + outer(k,v)
z' = z + k
read_vector = ((q @ M') / (sum(q*z') + 1e-6)) @ Wo
```

Initial z is nonnegative; validation rejects negative normalizer state. The
sequence path uses additive cumulative sums. A numerical zero input/message
still updates z and observation/history counts; it is not an absent fiber.

## Gated DeltaRule

Project q,k,v; L2-normalize q/k with norm clamped below at `1e-6`. Let
`beta=sigmoid(h w_beta)` and `a=sigmoid(h w_decay)`:

```
D  = a * M
M' = D + beta * outer(k, v - k @ D)
read_vector = (q @ M') @ Wo
```

The independent sequence contract rewrites this as `M'=A@M+B`, where
`A=a*(I-beta*outer(k,k))`, `B=beta*outer(k,v)`, and composes affine transforms
with an associative matrix scan. This is an algebraic correctness anchor.
It materializes dense transforms and can cost O(T log(T) d^3), compared with
O(T d^2) for the literal recurrence. It is not a high-performance DeltaNet
kernel. Use `prefill=False` to explicitly choose causal state steps while
retaining packed Full. A structured chunk kernel and measurements remain work.

Both profiles support native batch operations, node workers and exact sequence
contracts; Python scalar recurrence and Python sequence paths are independent
formula checks. Validation includes all state slots and their initial-state VJPs,
mixed-module graphs, clear, cyclic streaming and SettleGraph embedding.
