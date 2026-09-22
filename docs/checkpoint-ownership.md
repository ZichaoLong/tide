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
multi-graph application. Loading starts a new autograd segment. The separate two-clock bundle is described in `token-application-checkpoint.md`;
its CPU qualification is in `evidence/token-checkpoint-coordinates.md`.
## Standalone C++ owner and optimizer layer

The native core now has a standalone named-owner layer before persistence. A
`ParameterRegistry` can register explicit auxiliary tensors or collect the
trainable tensors in a `Model` under `nodes.*`, `regions.*` and scale names.
One registry can collect several graph models with prefixes; TensorImpl
identity preserves aliases across nodes and graphs while distinct tensors that
overlap storage remain distinct owners. Canonical names are the
lexicographically first aliases and optimizer groups retain their declared
order after alias normalization.

`SGD` and `AdamW` update only owners in their groups. An undefined gradient is
skipped, while a connected zero gradient still creates or updates optimizer
state, including decoupled AdamW decay. Shared aliases are updated once, and
`zero_grad` only clears the optimizer's own groups. The implementation is
independent of Python (`cpp/include/tide/parameters.h` and
`cpp/include/tide/optimizer.h`); `cpp/src/parameters.cpp` and
`cpp/src/optimizer.cpp` provide the core, with a small standalone executable
and `tests/test_cpp_optimizer.py` comparing both dtypes directly with
PyTorch's SGD/AdamW.

## Standalone native value checkpoints

The independent value layer uses the self-describing `TIDENCK1` schema version
1 in `cpp/include/tide/checkpoint.h` and the separate codec, preflight and
publication sources. A file records a caller-supplied graph identity, the
complete canonical alias partition, CPU FP32/FP64 dense owner values, and
optionally the built-in `torch.optim.sgd.SGD` or `torch.optim.adamw.AdamW`
class, ordered canonical owner groups, group options and named optimizer
slots. Tensor bytes are little-endian IEEE values and the payload has an
FNV-1a checksum. The format is deliberately independent of Python's torch
serialization; no Python-file interoperability claim follows from it.

Load reads and fully decodes the file before mutation. It checks identity when
an expected identity is supplied, alias topology, owner shapes/dtypes,
non-overlapping destination storage, optimizer class/order/options/state and
finite values. A failed check leaves live owners and optimizer state unchanged.
Publication writes and fsyncs a same-directory temporary file, creates the
final path with an exclusive hard link, removes the staging name and fsyncs
the directory. Existing targets are never overwritten. Omitting the optimizer
restores weights only; this format does not contain graph continuation, RNG,
data cursors or a training controller.

The standalone executable `tidegraph-checkpoint-check` and
`tests/test_cpp_checkpoint.py` cover both CPU dtypes, SGD/AdamW moments,
shared aliases, undefined versus connected-zero gradients, identity and
topology rejection, checksum/truncation preflight, unchanged-on-failure and
exclusive publication. The Python functions are only a client adapter;
`Checkpoint::save/load` remains usable from independent C++ code.

Implementation: `python/tidegraph/checkpoint.py` encodes and validates graph
state; `checkpoint_ownership.py` handles Python parameter and optimizer
identities. The independent native owner/step layer is in
`cpp/include/tide/parameters.h`, `cpp/include/tide/optimizer.h`,
`cpp/src/parameters.cpp` and `cpp/src/optimizer.cpp`.
`tests/test_checkpoint_ownership.py` covers the silent-reorder reproducer,
rejection without mutation, shared subset ownership and next-update equality.
`tests/test_cpp_optimizer.py` and the `tidegraph-optimizer-check` executable
cover native alias partitions, group order, None versus connected-zero
gradients and direct LibTorch/PyTorch update equality. Existing
module/executor checkpoint tests supply their numerical continuations.

## Exclusive atomic publication

Save serializes into a private temporary file in the target directory, flushes
and fsyncs it, atomically creates the final name with a no-overwrite hard link,
removes the staging name, then fsyncs the directory. Readers never see a partially
serialized final target. Existing files (including a competing writer that wins
during serialization) are preserved. Serialization or file-fsync failure removes
the temporary file and permits a retry at the same target path. The value schema
and load semantics do not change.

A process killed before cleanup may leave an unreferenced staging file. If the
final directory fsync fails after publication, save reports the error while the
complete target may already exist; inspect it rather than blindly overwriting.
The filesystem must support same-directory hard links and directory fsync; an
unsupported operation fails explicitly. This is the Linux CPU IO contract.

The prior direct-write failure is retained in
artifacts/checkpoint-partial-write-repro/: injected ENOSPC left an 18-byte final
file that could not load. tests/test_checkpoint_io.py injects serialization and
file-fsync failures, retry, an existing/racing writer and target visibility while
serialization is in progress. Implementation: checkpoint_io.py.
