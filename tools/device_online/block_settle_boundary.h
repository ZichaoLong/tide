#pragma once
#include "block_settle_flow.h"
namespace tide::device_online {
// Actual complete-cut metadata, independent of differentiable raw fields.
struct BlockSettleBoundary {
  at::Tensor node_present; // bool [B,nodes].
  at::Tensor region_times; // int64 [B,regions], -1 means no actual query.
  at::Tensor input_counts,input_times; // int64 [B,ports], initial occurrence -1.
};
BlockSettleBoundary append_block_settle_boundary(DeviceProgram&,const SettleGraph&,
  const BlockSettleFlowTape&,const at::Tensor& external_present,const BlockSettleBoundary&,
  int64_t tensor_budget);
}
