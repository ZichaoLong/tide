#pragma once
#include "reverse_links.h"

namespace tide::device_online {
// Append only normalized rows. The sum payload path handles kind0 separately.
// The caller zeroes out before the reverse-stage loop; this component adds
// per-event Jacobians in stable order, including missing all-softmax slots.
void append_aggregate_vjp(DeviceProgram&,const ReverseTape&,const ReverseLinks&,
    const at::Tensor& stage_count,const at::Tensor& stage_range,
    const at::Tensor& content_gradient,const at::Tensor& content_connected,
    const at::Tensor& messages,const at::Tensor& scale_partials,AggregateVjp& out,
    const at::Tensor& error,int64_t chunk_rows,int64_t tensor_budget_bytes);
} // namespace tide::device_online
