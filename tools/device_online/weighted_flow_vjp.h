#pragma once
// Complete local stage chain for the pinned weighted functions. It consumes
// actual forward tapes; graph ancestry and retained-window ownership are separate.
#include "weighted_flow_internal.h"
#include "weighted_parameter_vjp.h"
namespace tide::device_online {
using WeightedFieldAdjoints=std::map<std::string,WeightedRowAdjoint>;
struct WeightedRawAdjoint {at::Tensor value,connected;}; // [rows,width], [rows,fields]
struct WeightedNodeRoots {
  WeightedRowAdjoint content,full;
  WeightedRawAdjoint proposal;
  WeightedFieldAdjoints describe;
};
struct WeightedNodeGradient {
  WeightedRawAdjoint old,initial;
  at::Tensor payload,connected; // Physical input atoms, independently connected.
  std::vector<WeightedParameterPartial> parameters;
};
struct WeightedRegionAdjoint {
  std::map<int64_t,WeightedFieldAdjoints> describe;
  std::vector<WeightedParameterPartial> parameters;
};
WeightedNodeGradient append_weighted_node_vjp(DeviceProgram&,TaskServices&,const WeightedFlowNode&,
  const std::string& read_mode,const WeightedNodeRoots&,const at::Tensor& error,const WeightedFlowBudget&);
WeightedRegionAdjoint append_weighted_flow_region_vjp(DeviceProgram&,TaskServices&,const WeightedFlowRegion&,
  const WeightedFlowNodes&,const std::map<int64_t,WeightedRowAdjoint>& controls,
  const at::Tensor& error,const WeightedFlowBudget&);
WeightedRegionAdjoint append_weighted_region_views_vjp(DeviceProgram&,TaskServices&,const WeightedFlowRegion&,
  const std::vector<WeightedFlowView>&,const std::map<int64_t,WeightedRowAdjoint>& controls,
  const at::Tensor& error,const WeightedFlowBudget&);
namespace weighted_reverse {
WeightedRowAdjoint zeros(DeviceProgram&,const at::Tensor& shape);
WeightedRawAdjoint zeros(DeviceProgram&,const WeightedRawLayout&,const at::Tensor& shape);
// Adds only connected, present rows; absent cotangents may contain NaNs.
void add(DeviceProgram&,const WeightedRowAdjoint& target,const WeightedRowAdjoint& source,
  const at::Tensor& owners,const at::Tensor& error);
void add(DeviceProgram&,const WeightedRawLayout&,const WeightedRawAdjoint& target,
  const WeightedRawAdjoint& source,const at::Tensor& error);
void parameter(DeviceProgram&,std::vector<WeightedParameterPartial>&,const WeightedBanks&,
  const std::string& role,const at::Tensor& value,const at::Tensor& connected);
void parameters(DeviceProgram&,std::vector<WeightedParameterPartial>&,const WeightedBanks&,
  const std::string& prefix,const WeightedFieldAdjoints&);
void source(DeviceProgram&,TaskServices&,const WeightedFlowNode&,const WeightedAggregateGradient&,
  std::vector<WeightedParameterPartial>&,const at::Tensor& error,const WeightedFlowBudget&);
}
}
