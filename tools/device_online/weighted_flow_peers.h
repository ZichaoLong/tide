#pragma once
// One sample's node owners, not independent sample replicas. All decisions
// within a window are device requests, responses and a common commit.
#include "weighted_flow.h"
#include "weighted_flow_views.h"
#include "device_program.h"
#include <functional>
namespace tide::device_online {
struct WeightedFlowNode;
struct WeightedNodeInput;
class WeightedFlowPeers {
 public:
  WeightedFlowPeers(at::Device,const WeightedFlowLimits&);
  ~WeightedFlowPeers();
  using Nodes=std::vector<std::unique_ptr<WeightedFlowNode>>;
  using Budget=std::function<int64_t(DeviceProgram&)>;
  void bind(const Graph&,Nodes&,const Budget&);
  void append_begin(DeviceProgram&,const at::Tensor& error);
  void append_local_send(DeviceProgram&,const Graph&,Nodes&,const std::vector<WeightedNodeInput>&,
                         std::vector<WeightedFlowView>&,const at::Tensor&,const Budget&);
  void append_local_receive(DeviceProgram&,const at::Tensor& error);
  void append_full(DeviceProgram&,Nodes&,std::vector<WeightedFlowView>&,const at::Tensor&,const Budget&);
  void append_commit(DeviceProgram&,Nodes&,const at::Tensor& error);
  void append_stop(DeviceProgram&);
  void finish();
  void synchronize_inputs();
  void submit();void wait();void close();
  std::vector<const DeviceProgram*> programs()const;
  std::vector<at::Tensor> tensors()const;
  at::Tensor boundary_integer(int64_t node)const;
  at::Tensor boundary_present(int64_t node)const;
  int64_t size()const;
  int64_t workspace_bytes()const;
  int64_t packet_bytes()const;
 private:
  struct Impl;std::unique_ptr<Impl> impl_;
};
}
