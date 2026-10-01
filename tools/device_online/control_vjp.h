#pragma once
#include "state_vjp.h"
#include "tide/types.h"

namespace tide::device_online {
// Frozen actual forward values, not a replay of an independent reference.
// mode: 0 HARD, 1 HST, 2 SOFTP. Identity boundary nodes bypass Emit entirely.
struct ControlTape {
  at::Tensor read,raw_full;
  int64_t mode=0;
  double zeta=1.;
};
struct ControlVjp {
  at::Tensor fresh,events,connected,read,read_connected;
};
// Returns the fresh-Full cotangent, direct content/Read cotangents, and physical
// Read parameter partials. The complete candidate frame participates in softmax,
// even when only one selected node has a connected output. No discrete VJP.
ControlVjp append_control_vjp(CannProgram&,const Graph&,const StateTape&,const ControlTape&,
    const at::Tensor& stage_count,const at::Tensor& range,const at::Tensor& gradient,
    const at::Tensor& connected,const at::Tensor& error,int64_t tensor_budget_bytes);
// Whole-region Emit/softmax derivatives, without accessing a Read parameter
// bank. Compact owners consume scores/read_connected for their local Read VJP.
struct ControlScores {ControlVjp emit;at::Tensor scores,read_connected;};
ControlScores append_control_scores(CannProgram&,const Graph&,const StateTape&,const ControlTape&,
    at::ScalarType payload,const at::Tensor& stage_count,const at::Tensor& range,
    const at::Tensor& gradient,const at::Tensor& connected,const at::Tensor& error,int64_t tensor_budget);
void append_control_merge(CannProgram&,const ControlVjp&,const StateCotangents&,const at::Tensor& error);
void append_connection_union(CannProgram&,const at::Tensor& source,const at::Tensor& target,const at::Tensor& error);
} // namespace tide::device_online
