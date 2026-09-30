#pragma once
#include "cann_program.h"
#include "queue_closure.h"

namespace tide::device_online {
// Selected, computed Full outputs only. Each row is one node action:
// int64(sample,node,send_time,emission_position), payload, independent valid bit.
struct ActionBatch {at::Tensor coordinates,values,valid;};
// Broadcast Full delivery contract: outgoing edges receive the row times their
// physical edge scale. Static CSR stores topology only, never an event trace.
// Sparse/per-slot Full programs require a different delivery contract.
class BroadcastRouter {
 public:
  BroadcastRouter(int64_t nodes,int64_t samples,const std::vector<Wire>&,
                  int64_t capacity,at::Device);
  // sticky_error is shared with the transaction: 1 capacity, 2 bad metadata,
  // 3 actual message-time overflow. Outputs commit only after validation.
  AtomBatch append_stage(CannProgram&,const ActionBatch&,const at::Tensor& edge_scales,
                         const at::Tensor& sticky_error) const;
 private:
  int64_t nodes_,samples_,edges_,capacity_;
  at::Device device_;
  at::Tensor offsets_,edges_by_source_,targets_,delays_;
};
} // namespace tide::device_online
