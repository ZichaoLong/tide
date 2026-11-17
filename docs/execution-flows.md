# Execution

New model implementations own their tensor layouts and evaluation order.
Batch B, sequence positions and compatible nodes across operations. Real sparse
Full selection must not become complete dense Full followed by masking.
Flash attention, ordinary batched attention and chunked attention are chosen for
the actual family, shape, dtype and device. Preserve weighted-prior gradients,
original-time visibility and softmax normalization across key chunks.

CPU/NPU are the immediate platforms. PyTorch is the first public entry;
LibTorch/custom kernels are used for demonstrated scheduling/operator benefit.
Historical mixed-a/b/c and resident frameworks impose no layer or delivery duty.
Training includes preparation, forward, loss, backward, optimizer and sync.

Use all available CPU/memory/idle NPUs as useful without self-contention. Shared
NPU admission follows conversation authorization. Long work uses frozen inputs,
a detached background.slice systemd user service, bounded runtime/memory, logs
and checked terminal status. Never alter an active job's source.

Only minimal configuration, commands/results and reusable lessons are maintained.
No obligation to produce full action/dependency records or universal decoders.

The optional shared NPU adapter in scripts/npu_shared_queue.py requires an
expiring policy created from current observed device occupants and explicit
conversation authorization. If an existing cooperative exclusive lock is
borrowed, the policy also binds its actual PID/start/UID/device/inode. The old
lock is never rewritten or released. A separate whole-group borrower mutex,
the existing FIFO/two sampling phases, an additional post-lock resource check
and a lifetime identity/expiry monitor remain in force. Identity changes stop
our adapter/child, never the original owner. These measurements are shared
observations, not exclusive hardware timings. Policy files belong in ignored
run records and must not become a later conversation's standing permission.
