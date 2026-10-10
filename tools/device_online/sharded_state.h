#pragma once
#include "full_placement.h"
#include "remote_state.h"

namespace tide::device_online {
// Explicit internal placement. Whole-model training/public owner follows the
// state/cache reverse integration; this forward increment cannot certify it.
struct ModelPlacement {FullPlacement full,state;};
class ShardedState {
 public:
  static long double minimum_bytes(const ContentProfile&,const FullPlacement&,const Continuation&,const ContentLimits&);
  ShardedState(const ContentProfile&,FullPlacement,const Continuation&,at::Device,ContentLimits,int64_t budget);
  ~ShardedState();
  void append_read(DeviceProgram&,const ReadyBatch&,const ContentBatch&,const at::Tensor& stage,const at::Tensor& error,int64_t operator_budget);
  ContentUpdate append_update(DeviceProgram&,const ReadyBatch&,const ContentBatch&,const SelectionProposal&,
                              const at::Tensor& stage,const at::Tensor& event_count,const at::Tensor& error);
  void append_commit(DeviceProgram&,const at::Tensor& error);
  void append_stop(DeviceProgram&);
  void reset_window();void synchronize_inputs() const;void submit();void wait();void close();
  void export_states(Continuation&) const;void export_trace(std::vector<Event>&) const;
  int64_t reserved_bytes() const;int64_t program_count() const;int64_t workspace_bytes() const;int64_t packet_bytes() const;
  std::map<std::string,int64_t> stats() const;
  // After all programs complete; synchronization belongs only to error reporting.
  std::map<std::string,int64_t> failure_stats() const;
  std::vector<StateOwnerTape> reverse_parameters(int64_t tensor_budget) const;
  std::vector<StateOwnerBanks> parameter_banks() const;
  std::vector<StateOwnerValues> state_values() const;
  std::vector<ContinuationRows> continuation_rows() const;
  std::vector<Tensor> continuation_tensors() const;
 private:
  struct Impl;std::unique_ptr<Impl> impl_;
};
} // namespace tide::device_online
