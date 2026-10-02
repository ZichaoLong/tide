#pragma once
#include "state_reverse_pack.h"

namespace tide::device_online {
// Local stage rows are contiguous in the compact whole-window journal. All
// dynamic row indices derive from the candidate's actual device stage range.
struct StateReverseStage {
  StateTape tape;
  at::Tensor range,destinations,score_gradient,read_connected;
  StateCotangents cot;
};
StateReverseStage append_state_reverse_stage(CannProgram&,const StateReversePacket&,
    const StateTape& parameters,const at::Tensor& global_range,const StateCotangents&,
    const at::Tensor& node_ids,const at::Tensor& score_gradient,const at::Tensor& read_connected,
    const at::Tensor& error,int64_t tensor_budget,
    const ReverseGatherInput& events,const ReverseGatherInput& connected);
} // namespace tide::device_online
