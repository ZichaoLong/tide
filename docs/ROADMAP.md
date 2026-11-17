# Follow-up after the current instance delivery

The agreed fresh FC/HB implementation and finite CPU/NPU FP32 delivery are
complete at the scope in [semantics](semantics.md), with measured qualifications
and limits in [the report](evidence/instances-fp32-20261118.md). No job is active.
These are remaining limits/opportunities, not an instruction to restart a broad
rewrite or resume old P1–P6.

1. Preserve four whole-model NPU finite-deviation records. Their support,
   metadata, gradient connectivity, finite updates and exact restore passed;
   the worst case has a bounded FP64 explanation. Strict elementwise passage
   of every FP32 state/value remains unclaimed. Revisit if a consumer's actual
   scale, learning behavior or numerical requirement makes it material.
   Actual resumed FP32 AdamW updates also retain small common-bias parameter
   deviations despite exact saved data/forward; bitwise optimizer replay is
   not promised. Their attempts and thresholds remain in the same report.
2. FC Full packing currently processes 2.02× selected rows in the measured case;
   consider query buckets/ragged operators. Lattice's factor is only 1.07×.
   Head-query and gate/up packing, short Attention backend choice and Shannon
   iteration cost are other concrete candidates, to prioritize from actual need.
   KV time ordering currently invokes device AiCPU Argsort; consider stable
   compaction while retaining int64 time semantics.
3. Larger/longer configurations and billion-parameter capacity/performance are
   unmeasured. Choose useful downstream configurations, with real activity and
   padding counters; do not infer 3% Full from k/n or exhaust all combinations.
4. Native/LibTorch kernels may be added for demonstrated benefit. CUDA,
   multi-device, exclusive NPU timings and resident scheduling are not qualified
   by this delivery and are not inherited from historical implementations.

All current models/configuration candidates remain available in PyTorch. No
FP16/BF16 work, old teaching binding, LH-DAG/PDG or arbitrary graph engine is
required. New downstream work starts from the current public API/configuration,
fixed mathematical source and observed needs. Relevant checks and portable
commands are in the report; [measured lessons](performance-lessons-instances.md)
explain successful changes and remaining costs.
