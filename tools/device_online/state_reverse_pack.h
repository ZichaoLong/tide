#pragma once
#include "state_reverse_view.h"

namespace tide::device_online {
struct StateReversePacket {
  at::Tensor event_meta,event_values,event_count,fiber_meta,fiber_values,fiber_count;
  // Only local consumer links and per-atom Aggregate scales are populated.
  // Global routing/producer/physical-scale reduction remains on the coordinator.
  ReverseLinks links;
  at::Tensor event_rows,fiber_rows; // local row -> original coordinator row
};
StateReversePacket append_state_reverse_pack(CannProgram&,const ReverseTape&,const ReverseLinks&,
    const at::Tensor& node_mapping,int64_t local_nodes,int64_t event_capacity,int64_t fiber_capacity,
    const at::Tensor& error,int64_t tensor_budget);
} // namespace tide::device_online
