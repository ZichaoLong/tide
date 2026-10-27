#pragma once
#include "weighted_raw_state.h"
#include "device_program.h"
namespace tide::device_online {
struct WeightedRawUpdateTape {
  WeightedRawLayout layout;
  WeightedRawState result;
  at::Tensor old_integer,proposal,fields,mapping,receive,written;
};
struct WeightedRawUpdateGradient {
  at::Tensor old_values,proposal,mass,old_connected,proposal_connected,mass_connected;
};
// proposal is in the raw floating layout: fixed post-update fields and one
// unweighted entry at each ragged field's start. KV weights come from mass.
// This layer owns raw retention/clock/cache; memory formulas supply fixed fields.
// Primal storage/mass/proposal are all FP32 or all FP16. FP16 uses vendor casts
// around FP32 raw arithmetic; actual saved operands and all adjoints are FP32.
// Casts have the declared identity VJP, independent of rounding staircases.
WeightedRawUpdateTape append_weighted_raw_update(DeviceProgram&,const WeightedRawLayout&,
  const WeightedRawState&,const at::Tensor& time,const at::Tensor& mass,const at::Tensor& present,
  const at::Tensor& proposal,const at::Tensor& error,int64_t budget=256*1024*1024);
// Independent roots per row and named floating field, including adapter value.
WeightedRawUpdateGradient append_weighted_raw_update_vjp(DeviceProgram&,const WeightedRawUpdateTape&,
  const at::Tensor& gradient,const at::Tensor& roots,const at::Tensor& error,int64_t budget=256*1024*1024);
// Explicit guarded commit into a live private owner, after all local decisions.
void append_weighted_raw_adopt(DeviceProgram&,const WeightedRawState& target,
  const WeightedRawState& proposal,const at::Tensor& error);
}
