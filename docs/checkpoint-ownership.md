# Parameter and optimizer ownership at a value checkpoint

The current checkpoint schema is defined in `semantics.md`. It binds an
optimizer's positional state IDs to named parameter owners before restoring
weights. A plain PyTorch optimizer state dict does not do this: swapping two
same-shaped parameters silently exchanges their momentum or Adam statistics.

Each Parameter object has one canonical owner, the lexicographically first of
its registered names. All other names are aliases. Save records the complete
alias partition, the optimizer class and the ordered canonical names in each
parameter group. Subset optimizers are supported. Duplicate parameters and
parameters outside the supplied model are rejected. Physical tensor storage
overlap between distinct Parameter objects is not declared parameter sharing.

Load requires the reconstructed model to have the same alias partition and the
optimizer to have exactly the same class, group membership and parameter order.
It does not guess a remapping. Group options, including learning rates, restore
from the checkpoint. Omitting the optimizer restores weights/continuation only.
Old schema versions are rejected; v4 cannot establish optimizer ownership.

Before changing live weights or optimizer state, load checks:

- Graph identity and complete continuation, tensor keys/shapes/dtypes/finiteness.
- Agreement of all saved values belonging to one shared Parameter.
- Optimizer ownership, canonical state IDs, group cardinality and finite state.
- SGD momentum and Adam/AdamW moment shapes/dtypes, step and AMSGrad presence.
- Standard optimizer `load_state_dict` on a temporary deep copy.

The temporary load adds checkpoint-time memory/copy cost. Ordinary built-in
optimizer layouts are qualified individually; arbitrary custom optimizer or
module hooks with external side effects do not have a transactional guarantee.
This is a single graph/model value checkpoint. It does not restore a live
autograd graph, `.grad` buffers, RNG, data cursor, training controller or a
multi-graph application. Loading starts a new autograd segment. Cross-graph
parameter ownership and composite checkpointing remain in `ROADMAP.md`.

Implementation: `python/tidegraph/checkpoint.py` encodes and validates graph
state; `checkpoint_ownership.py` handles parameter and optimizer identities.
`tests/test_checkpoint_ownership.py` covers the silent-reorder reproducer,
rejection without mutation, shared subset ownership and next-update equality.
Existing module/executor checkpoint tests supply their numerical continuations.
