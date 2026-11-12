#pragma once
#include "block_settle_session.h"
namespace tide::device_online {
struct BlockSettleModel {
  SettleGraph graph;
  std::vector<BlockSettleKVBinding> bindings;
  ParameterRegistry registry;
  std::vector<BlockKVParameterRoles> roles;
  std::vector<WeightedRawState> initial,bound_initial;
};
// Static capability/lowering only. It never executes a CPU graph trajectory.
// Failure is explicit; a caller may select a general executor before execution.
BlockSettleModel lower_block_settle_model(const SettleGraph&,const Model&,
  const Continuation&,int64_t cache_capacity,int64_t binding_budget);
}
