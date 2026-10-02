# Model-specific attention gradient admission

Qualified source `c3ed0f26dfc59f3e4ae3b8c032a53fe52cc8bca2`.
The [audited record](consumer-add-capacity-20261002.json) pins the installed
consumer, unchanged shared-packet backend, clean tests and observations.
Audit: `TASK/launchers/add_capacity_evidence.py SOURCE`.

The Add consumer was charged for per-node QKV/output matrix gradients that exist
only in the Attention consumer. Capacity admission now conditions that charge on
the declared model profile and reports `attention_parameter_gradients` as a
subtotal of physical/canonical gradients. Real edge projections, scalar
Aggregate parameters, vector LH/state/Read adjoints, canonical gradients and
optimizer storage retain their charges. Safety margins and logical capacities
are unchanged. This does not alter execution or numerical results.

Clean **CPU 10 / NPU 9 checks passed**, without skips. The CPU checks include an
actual materialized Add/Attention model inventory, C++/Python planner agreement,
heterogeneous budgets, retention, precision, overflow and bounded automatic
sample admission. Eight complete NPU update trajectories cover both consumers,
both model profiles and FP32/FP16 against independent CPU execution; the ninth
checks refusal before model construction. All allocator observations remain
within admission. Core, runtime library and device kernels were unchanged.

An offline original Add D2048/B512/T12/V50304 plan with ten cards and physical B2
groups admits a maximum 52.012 GiB after retaining the original 60 GiB/card cap
and safety margin. It uses explicit queue/arrival capacity 896, output 64, trace
3072 and unchanged KV256/KV-trace8192. This is a plan, not an executed full-size
training result. A bounded original-width B4 pilot and original B512 task are
prepared; a shortage of free cards did not trigger a new waiting job.
