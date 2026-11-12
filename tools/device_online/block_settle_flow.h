#pragma once
#include "block_kv_local.h"
#include "task_services.h"
namespace tide::device_online {
// Explicit concrete block bindings. Parameter classification never deduplicates
// physical atoms; input/edge mappings contain one owner per physical identity.
struct BlockSettleKVBinding {
  weighted::KVConfig memory;
  weighted::RegionalConfig regional;
  std::string read_mode="+";
  double aggregate_alpha=0; // Configuration identity; FP32 numeric conversion is checked.
  BlockKVLocalParameters parameters;
  std::vector<int64_t> node_owners,input_owners,edge_owners;
};
struct BlockSettleFlowInput {
  at::Tensor positions; // int64 [B,L], increasing independently per sample.
  at::Tensor payload,present; // [B,L,input_ports,d+1], bool [B,L,input_ports].
  std::vector<WeightedRawState> initial,bound_initial; // Region-local B*N rows.
};
struct BlockPhysicalColumns {
  at::Tensor payload,present,times; // physical input/edge/output identities kept.
};
struct BlockSettleRegionTape {
  std::shared_ptr<BlockKVLocalTape> local;
  at::Tensor nodes,atoms,atom_counts,physical_atoms,source_mapping;
};
struct BlockSettleFlowTape {
  std::shared_ptr<BlockDispatcher> dispatcher;
  std::vector<BlockSettleRegionTape> regions;
  BlockPhysicalColumns nodes,edges,outputs;
  at::Tensor positions;
};
// One device choose/commit loop. A task is a whole complete-region B x L block,
// including its empty-fiber state identity. Numerical activity stays per node
// and position. The coordinator calls bounded whole-region numerical programs
// through device notifications; no actual-trajectory host prepass or token loop.
// First concrete lowerer: aligned Settle ports, homogeneous KV shape per region.
// Returned final states are staging. The caller publishes only after success.
BlockSettleFlowTape append_block_settle_flow(DeviceProgram&,TaskServices&,const SettleGraph&,
  const std::vector<BlockSettleKVBinding>&,const BlockSettleFlowInput&,
  int64_t tensor_budget);
}
