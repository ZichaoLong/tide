#pragma once
// Independent small graph formula: literal local equations and scalar causal
// regions. This oracle never invokes candidate block kernels or its scheduler.
#include "block_settle_parameters.h"
namespace tide::device_online::block_flow_check {
constexpr int64_t Batch=2,Length=2,Nodes=3,Frames=Batch*Length,Capacity=6;
constexpr int64_t Start=(int64_t{1}<<53)+17;
using Fields=std::map<std::string,at::Tensor>;
struct Fixture {
  SettleGraph graph;
  std::vector<BlockSettleKVBinding> bindings;
  BlockSettleFlowInput input;
  ParameterRegistry registry;
  std::vector<BlockKVParameterRoles> roles;
  std::vector<std::vector<State>> states,bound;
  Fixture();
};
struct Label {std::string kind,name;int64_t region=0,row=0;};
struct Formula {
  std::vector<at::Tensor> leaves;
  std::vector<Label> labels;
  std::vector<std::vector<State>> old,proposal,final;
  std::vector<std::vector<Fields>> describe;
  std::vector<std::vector<at::Tensor>> received,memory,total,controls,full;
  std::vector<at::Tensor> messages;
  at::Tensor present,times;
  std::vector<std::vector<State>> bound;
  Fields canonical;
  int64_t windows=1;
};
at::Tensor values(at::IntArrayRef,float shift);
Formula formula(const Fixture&,const Formula* previous=nullptr);
std::vector<BlockSettleKVBinding> device_bindings(const Fixture&,at::Device);
BlockSettleFlowInput device_input(const Fixture&,at::Device);
}
