#pragma once
#include "packed_fiber_attention.h"
#include "packed_event_attention.h"
#include "state_owner_tape.h"

namespace tide::device_online {
struct StateReadProposal {
  at::Tensor values;
  FiberStage fiber;
  EventAttentionStage event;
};
// An actual compact state/Read/KV owner. It knows no global selection policy,
// routes or input event schedule. Its caller supplies current device-ready work.
class StateOwner {
 public:
  static long double minimum_bytes(const ContentProfile&,const std::vector<int64_t>&,
                                   const Continuation&,const ContentLimits&);
  StateOwner(const ContentProfile&,std::vector<int64_t>,const Continuation&,at::Device,
             ContentLimits,int64_t tensor_budget);
  StateReadProposal append_read(CannProgram&,const ReadyBatch&,const ContentBatch&,const at::Tensor& error);
  ContentUpdate append_update(CannProgram&,const ReadyBatch&,const ContentBatch&,const SelectionProposal&,
      const StateReadProposal&,const at::Tensor& stage,const at::Tensor& error);
  void append_commit(CannProgram&,const ContentUpdate&,const StateReadProposal&,const SelectionProposal&,
                     const at::Tensor& error);
  void reset_window();
  void export_states(Continuation&) const;
  void export_trace(std::vector<Event>&) const;
  const StateKernelProfile& profile() const {return profile_;}
  const ContentState& state() const {return state_;}
  const std::vector<int64_t>& global_nodes() const {return global_nodes_;}
  int64_t reserved_bytes() const {return reserved_;}
  std::map<std::string,int64_t> stats() const;
  int64_t reverse_parameter_bytes() const;
  StateOwnerTape reverse_parameters(int64_t tensor_budget) const;
  StateOwnerBanks parameter_banks() const;
  std::vector<ContinuationRows> continuation_rows() const {
    std::vector<ContinuationRows> out;
    if(fiber_){auto xs=fiber_->continuation_rows();out.insert(out.end(),xs.begin(),xs.end());}
    if(event_){auto xs=event_->continuation_rows();out.insert(out.end(),xs.begin(),xs.end());}
    return out;
  }
  std::vector<Tensor> continuation_tensors() const {
    std::vector<Tensor> out{state_.values,state_.clocks,state_.present};
    if(fiber_){auto xs=fiber_->continuation_tensors();out.insert(out.end(),xs.begin(),xs.end());}
    if(event_){auto xs=event_->continuation_tensors();out.insert(out.end(),xs.begin(),xs.end());}
    return out;
  }
 private:
  std::vector<int64_t> global_nodes_;
  StateKernelProfile profile_;
  ContentLimits limits_;
  ContentState state_;
  at::Tensor coefficients_,event_count_;
  std::unique_ptr<PackedFiberAttention> fiber_;
  std::unique_ptr<PackedEventAttention> event_;
  int64_t reserved_=0,physical_edges_=0;
};
} // namespace tide::device_online
