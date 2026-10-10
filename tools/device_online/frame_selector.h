#pragma once
#include "ready_batch.h"

namespace tide::device_online {
struct SelectionPolicy {int64_t budget;bool count_priority=true,positive_only=false;};
struct SelectionHistory {at::Tensor counts,seen,last_time,present;};
struct SelectionProposal {
  SelectionHistory history;
  at::Tensor active,controls,branch; // per-fiber bool/FP32; int32 any-frame flag
};
// count-v1 / positive-v1 region contracts. Descriptors are explicit FP32;
// producing them from content/state is a separate numerical module contract.
class FrameSelector {
 public:
  FrameSelector(std::vector<int64_t> owners,std::vector<SelectionPolicy>,int64_t samples,
                at::Device,int64_t workspace_budget_bytes=64*1024*1024);
  SelectionHistory initial() const;
  SelectionProposal append_stage(DeviceProgram&,const ReadyBatch&,const at::Tensor& descriptors,
                                 const SelectionHistory&,const at::Tensor& error) const;
  // Place after every action that can refuse a proposal, including routing and
  // queue replacement. An error then preserves the entire original history.
  void append_commit(DeviceProgram&,const SelectionHistory&,const SelectionProposal&,
                     const at::Tensor& error) const;
 private:
  int64_t nodes_,regions_,samples_,budget_;
  at::Device device_;
  at::Tensor owner_,policies_;
};
} // namespace tide::device_online
