#pragma once
#include "device_program.h"
#include "queue_closure.h"

namespace tide::device_online {
// All lengths are device int64. Physical buffers are capacity-sized; only
// counts entries are logically present. The branch is int32[1], 0 empty/error.
struct ReadyBatch {
  AtomBatch atoms;                     // canonical stable atom order
  at::Tensor consumed;                 // int32 mask in original queue order
  at::Tensor fiber_offsets, fibers;    // (sample,node,time,frame index)
  at::Tensor frame_offsets, frame_fibers, frames; // (sample,region,time)
  at::Tensor counts, branch;           // [atoms,fibers,frames], [any]
};
class DeviceReady {
 public:
  DeviceReady(const std::vector<int64_t>& owners,int64_t regions,const std::vector<Wire>&,
              int64_t samples,at::Device,bool prefill,const std::vector<int64_t>& causal_regions={});
  // Complete sealed input through stop. Produces whole fibers AND complete
  // region-time candidate sets. Numerical state/Full contracts remain separate.
  ReadyBatch append_stage(DeviceProgram&,const AtomBatch&,const at::Tensor& stop,
                          const at::Tensor& sticky_error) const;
 private:
  int64_t nodes_,regions_,samples_;
  at::Device device_;
  bool prefill_;
  QueueClosure topology_;
  at::Tensor causal_regions_;
};
} // namespace tide::device_online
