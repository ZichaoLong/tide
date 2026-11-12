#pragma once
#include "block_settle_session.h"
namespace tide::device_online {
namespace block_session_detail {
struct Version {at::Tensor value;int64_t version;const void* data;bool trainable;};
void remember(std::vector<Version>&,const at::Tensor&);
void validate(const std::vector<Version>&);
std::string identity(const SettleGraph&,const std::vector<BlockSettleKVBinding>&,
  const ParameterRegistry&,const std::vector<BlockKVParameterRoles>&,const std::vector<WeightedRawState>&);
ParameterVjp parameter_layout(const ParameterRegistry&,at::Device,int64_t budget);
std::vector<at::Tensor> tensors(const BlockSettleFlowTape&);
}
struct BlockSettleSession::Impl {
  struct Window {BlockSettleWindow view;std::unique_ptr<DeviceProgram> program;std::vector<block_session_detail::Version> versions;};
  SettleGraph graph;std::vector<BlockSettleKVBinding> bindings;ParameterRegistry registry;
  std::vector<BlockKVParameterRoles> roles;at::Device device;BlockSettleLimits limits;
  DeviceOptimizerKind kind;std::unique_ptr<DeviceOptimizer> optimizer;
  ParameterVjp layout,gradient;std::vector<WeightedRawState> state,bound;
  at::Tensor positions;std::string fingerprint;int64_t generation=0,next_token=0;
  bool failed=false,ready=false,closed=false;
  std::vector<Window> windows;std::unique_ptr<DeviceProgram> reverse;
  std::vector<block_session_detail::Version> originals,gradient_versions;
  Impl(SettleGraph,std::vector<BlockSettleKVBinding>,const ParameterRegistry&,
    std::vector<BlockKVParameterRoles>,std::vector<WeightedRawState>,std::vector<WeightedRawState>,
    at::Device,DeviceOptimizerKind,std::vector<OptimizerGroup>,BlockSettleLimits,const BlockSettleCheckpoint*);
  void check()const;
  void run(DeviceProgram&,const at::Tensor&);
  void publish(DeviceProgram&);
  void discard();
  void budget(const DeviceProgram&)const;
};
}
