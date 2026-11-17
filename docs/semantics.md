# Adopted models

Fixed core-5 and local/instance revision: 4f5b9f20a4030d68ee66ae2d26e34659a5e37234.
See ../math/instances/backbone-comparison.md and the referenced new HB bindings.

FC, HB-Lattice and Exchange use Settle timing. Multiscale supports Settle/Timed.
K = KV/sum; S = SSD/state; X = D/KV and H/SSD (X equals K without hubs).
All use Read+, all positive-reception updates retained, encoded node descriptions,
regional self-attention and the chosen role quad/Shannon or ordinary pool.
Three sharing profiles preserve private node state. Original logical times govern
KV windows, positions and SSD convolution. k bounds mass, not active node count.

Correctness tests target actual values, support, parameter gradients and stateful
continuation. Full action journals and generic intermediate decoding are omitted.
The rewrite intentionally has a new public API and state representation.
