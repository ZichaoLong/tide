#pragma once
#include "reverse_links.h"

namespace tide::device_online {
// Static reverse metadata for a complete graph or an order-preserving node
// projection. Input/edge IDs stay physical and global; this is not a Graph.
struct StateReverseLayout {
  int64_t nodes,width,inputs,sources;
  std::vector<int64_t> source_counts,event_offsets,fiber_offsets;
  StateReverseLayout(const std::vector<Node>&,const std::vector<int64_t>& source_counts,
                     int64_t width,int64_t inputs,int64_t edges);
};
struct StateReverseView {
  StateReverseLayout layout;
  StateTape state;
  at::Tensor fiber_meta,fiber_values,sources;
  explicit StateReverseView(const ReverseTape&);
  StateReverseView(StateReverseLayout,StateTape,at::Tensor fiber_meta,at::Tensor fiber_values,at::Tensor sources);
};
} // namespace tide::device_online
