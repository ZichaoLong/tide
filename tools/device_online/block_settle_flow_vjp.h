#pragma once
#include "block_settle_flow.h"
namespace tide::device_online {
struct BlockPhysicalRoots {
  WeightedRowAdjoint strength,content; // Independent c and unscaled G roots.
};
struct BlockSettleFlowRoots {
  std::vector<BlockKVLocalRoots> regions;
  BlockPhysicalRoots edges,outputs;
};
struct BlockSettleFlowGradient {
  at::Tensor inputs,input_connected;
  std::vector<BlockKVLocalGradient> regions;
};
// Reverse consumes the actual forward journal. Future descendants and repeated
// physical edges contribute to the same source message once per actual use.
BlockSettleFlowGradient append_block_settle_flow_vjp(DeviceProgram&,const SettleGraph&,
  const BlockSettleFlowTape&,const BlockSettleFlowRoots&,int64_t tensor_budget);
}
