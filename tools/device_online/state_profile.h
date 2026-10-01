#pragma once
#include "tide/types.h"

namespace tide::device_online {
struct ContentProfile;
// A validated kernel view, not a Graph. Local node IDs are an order-preserving
// projection; physical input/edge IDs and logical source slots remain global.
// Never compile this view as a standalone graph or use it for queue closure.
struct StateKernelProfile {
  std::vector<Node> nodes;
  std::vector<NodeWeights> weights;
  std::vector<Region> regions;
  std::vector<int64_t> source_counts;
  int64_t width,input_count;
  at::ScalarType dtype;
  bool all_content;
  at::Tensor sources,read,read_modes,read_kinds,decay,retention,clock_policy,config;
  StateKernelProfile(const ContentProfile&);
  // Requires deferred CPU tables. Only owned state/Read parameters are uploaded.
  StateKernelProfile(const ContentProfile&,const std::vector<int64_t>& global_nodes,at::Device);
};
// Initial boundary projection only; neither events nor candidate decisions are
// supplied here. Export performs the inverse mapping at an explicit boundary.
Continuation state_shard_initial(const Continuation&,const std::vector<int64_t>& global_nodes);
} // namespace tide::device_online
