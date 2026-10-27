#pragma once
#include "device_program.h"
#include "queue_closure.h"
namespace tide::device_online {
// Independent roots of unscaled (c,G), indexed by the ACTUAL forward journal.
struct WeightedReverseEvents {
  at::Tensor times,present; // int64/bool [stage,node,sample].
  at::Tensor values,connected; // FP32 [stage,node,sample,payload], bool [stage,node,sample].
};
struct WeightedReverseBoundary {
  at::Tensor coordinates,present; // Exact left-pending + external atom identities.
  at::Tensor values,connected; // FP32 [capacity,payload], bool [capacity].
};
// One device action per reversed stage. Parallel edges add distinct uses to the
// producing Full; old pending/external values stay independently rooted. No
// trajectory or gradient leaves the device. Caller serializes uses and seeds
// accumulation buffers once per backward window. Sticky error invalidates the
// entire derivative result; no partial export is valid after refusal.
void append_weighted_reverse_route(DeviceProgram&,const AtomBatch&,const at::Tensor& indices,
  const at::Tensor& lengths,const at::Tensor& gradients,const at::Tensor& roots,
  const at::Tensor& edge_sources,const WeightedReverseEvents&,const WeightedReverseBoundary&,
  const at::Tensor& stage,const at::Tensor& begin_cut,const at::Tensor& error);
}
