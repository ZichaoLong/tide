#pragma once
#include "block_settle_parameters.h"
#include "block_settle_boundary.h"
#include "device_optimizer.h"
namespace tide::device_online {
struct BlockSettleLimits {
  // Per-call new retained forward/reverse tensors; optimizer persistent+step
  // tensors; total unique live tensors plus all retained program workspaces.
  int64_t forward_bytes=512LL<<20,backward_bytes=512LL<<20,optimizer_bytes=128LL<<20;
  int64_t total_bytes=2LL<<30,workspace_bytes=512LL<<20,windows=4;
  int32_t timeout_ms=30000;
};
struct BlockSettleCheckpoint {
  int64_t schema=2,generation=0,next_token=0;
  std::string identity;
  std::vector<std::vector<std::string>> aliases;
  std::map<std::string,at::Tensor> parameters;
  DeviceOptimizerKind optimizer=DeviceOptimizerKind::adamw;
  std::vector<OptimizerGroup> groups;
  ResidentOptimizerState slots;
  std::vector<WeightedRawState> state,bound_initial;
  at::Tensor last_positions; // int64 [B], independent sample cuts.
  BlockSettleBoundary boundary;
};
struct BlockSettleWindow {
  int64_t token=0,generation=0;
  std::shared_ptr<const BlockSettleFlowTape> tape;
  BlockSettleBoundary boundary;
};
struct BlockSettleAdjoints {
  std::vector<BlockSettleFlowGradient> windows;
  ParameterVjp parameters;
  std::vector<WeightedRowAdjoint> bound_initial; // Sum all uses of each fixed Read0 binding.
};
struct BlockSettleStorageUsage {int64_t tensor_bytes,workspace_bytes,optimizer_bytes;};
// New owner for the block executor; never wraps the earlier TaskServices flow.
// Caller canonical registry defines aliases; concrete bank rows do not become
// parameter identities. Boundary calls are synchronous, inner windows resident.
class BlockSettleSession {
 public:
  BlockSettleSession(SettleGraph,std::vector<BlockSettleKVBinding>,const ParameterRegistry&,
    std::vector<BlockKVParameterRoles>,std::vector<WeightedRawState> initial,
    std::vector<WeightedRawState> bound_initial,at::Device,DeviceOptimizerKind,
    std::vector<OptimizerGroup>,BlockSettleLimits = {},const BlockSettleCheckpoint* = nullptr,
    const BlockSettleBoundary* initial_metadata = nullptr,const at::Tensor& initial_positions = {});
  ~BlockSettleSession();
  BlockSettleWindow advance(const at::Tensor& positions,const at::Tensor& payload,const at::Tensor& present);
  // Roots are in forward window order; reverse carries exact raw adjoints
  // between windows and reduces every canonical parameter use once.
  BlockSettleAdjoints backward(const std::vector<BlockSettleFlowRoots>&);
  void step();
  void detach();
  BlockSettleCheckpoint checkpoint()const;
  BlockSettleStorageUsage storage_usage()const; // Host metadata; no numerical device read.
  std::map<std::string,int64_t> resource_usage()const; // Current live owners; not peak HBM/runtime internals.
  void close();
 private:
  struct Impl;std::unique_ptr<Impl> impl_;
};
// Standalone LibTorch serialization, CPU records only. Loader constructs no
// model and never evaluates reference mathematics to fill missing state.
void save_block_settle_checkpoint(const BlockSettleCheckpoint&,const std::string& path,
  int64_t max_bytes=int64_t{1}<<32);
BlockSettleCheckpoint load_block_settle_checkpoint(const std::string& path,
  int64_t max_bytes=int64_t{1}<<32);
}
