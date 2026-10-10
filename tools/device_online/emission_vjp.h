#pragma once
#include "reverse_links.h"
namespace tide::device_online {
struct EmissionReverse {
  EmissionVjp gradient;
  at::Tensor rows,parameters;
  int64_t chunk=0;
};
// Construct once before the reverse stage loop. All associations are built from
// this candidate's own selected-slot journal; no CPU event prepass is accepted.
EmissionReverse prepare_emission_reverse(DeviceProgram&,const ReverseTape&,const ReverseLinks&,
    const at::Tensor& error,int64_t budget,int64_t max_rows=1,int64_t operator_budget=0);
EmissionReverse prepare_emission_reverse(DeviceProgram&,const ReverseTape&,const ReverseLinks&,
    const at::Tensor& error,int64_t budget,int64_t max_rows,int64_t operator_budget,const std::vector<ProjectionGradient>& reuse);
// Called while constructing a device-controlled reverse stage, not at runtime
// on the host. Matrix work is bounded by chunk rows; reductions are ordered.
void append_emission_reverse(DeviceProgram&,const ReverseTape&,const ReverseLinks&,const EmissionReverse&,
    const at::Tensor& messages,const at::Tensor& connected,const at::Tensor& range,
    const at::Tensor& full_gradient,const at::Tensor& error,int64_t max_rows,int64_t budget);
} // namespace tide::device_online
