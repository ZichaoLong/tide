#pragma once
// Actual node results visible to the global region/queue owner. Physical node
// state and numerical tapes stay separate; see resident-weighted-placement.md.
#include "weighted_raw_state.h"
#include <map>
namespace tide::device_online {
struct WeightedFlowNode;
struct WeightedFlowView {
  at::Tensor present,times,content;
  WeightedRawState old,proposal;
  std::map<std::string,at::Tensor> describe;
  at::Tensor control,active,full,payload;
};
WeightedFlowView weighted_flow_view(const WeightedFlowNode&);
}
