#pragma once
#include "weighted_flow_vjp.h"
#include "weighted_reverse_route.h"
namespace tide::device_online {
// Internal resident backend contract. Public record/root binding and retained
// training tokens live above this actual-journal numerical owner.
struct WeightedWindowNodeRoots {
  WeightedRowAdjoint content,control,full;
  WeightedRawAdjoint old,proposal,final;
  WeightedFieldAdjoints describe;
};
struct WeightedWindowRoots {
  std::vector<WeightedWindowNodeRoots> nodes;
  WeightedRowAdjoint pending,messages,fibers;
};
class WeightedWindowAdjoint {
 public:
  explicit WeightedWindowAdjoint(WeightedFlow&);
  WeightedWindowAdjoint(WeightedFlow&,int64_t total_tensor_budget_bytes);
  ~WeightedWindowAdjoint();
  WeightedWindowAdjoint(const WeightedWindowAdjoint&)=delete;
  WeightedWindowAdjoint& operator=(const WeightedWindowAdjoint&)=delete;
  WeightedWindowRoots& roots(){return roots_;}
  const std::vector<WeightedRawAdjoint>& states()const{return left_;}
  const WeightedReverseBoundary& boundary()const{return boundary_;}
  const ParameterVjp& parameters()const{return parameters_;}
  const std::vector<std::unique_ptr<WeightedFlowNode>>& nodes()const{return nodes_;}
  std::map<std::string,at::Tensor> records()const;
  void run();
  void close();
 private:
  WeightedFlow& forward_;int64_t generation_,budget_;bool failed_=false;
  std::unique_ptr<DeviceProgram> program_;
  std::unique_ptr<TaskServices> services_;
  WeightedFlowNodes nodes_;
  std::vector<std::unique_ptr<WeightedFlowRegion>> regions_;
  WeightedWindowRoots roots_;
  std::vector<WeightedRawAdjoint> left_;
  WeightedReverseBoundary boundary_;
  ParameterVjp parameters_;
  at::Tensor error_,stage_;
  int64_t remaining(DeviceProgram&)const;
};
namespace weighted_window {
WeightedRowAdjoint row(const at::Tensor& shape,int64_t leading=2);
WeightedRawAdjoint raw(const WeightedRawLayout&,const at::Tensor& shape,int64_t leading=2);
at::Tensor load(DeviceProgram&,const at::Tensor& stage,const at::Tensor& source);
WeightedRowAdjoint load(DeviceProgram&,const at::Tensor& stage,const WeightedRowAdjoint&);
WeightedRawAdjoint load(DeviceProgram&,const at::Tensor& stage,const WeightedRawAdjoint&);
void route(DeviceProgram&,const AtomBatch&,const WeightedRowAdjoint&,const at::Tensor& edges,
  const WeightedReverseEvents&,const WeightedReverseBoundary&,const at::Tensor& stage,const at::Tensor& begin,const at::Tensor& error);
WeightedRawAdjoint mask(DeviceProgram&,const WeightedRawLayout&,const WeightedRawAdjoint&,
  const at::Tensor& present,const at::Tensor& error);
}
}
