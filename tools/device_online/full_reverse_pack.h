#pragma once
#include "full_vjp.h"
#include "reverse_gather.h"
namespace tide::device_online {
struct FullReverseBatch {
  FullTape tape;
  at::Tensor gradient,connected,destinations,branch;
};
FullReverseBatch append_full_reverse_pack(DeviceProgram&,const FullTape& stage,const at::Tensor& gradient,
    const at::Tensor& connected,const at::Tensor& mapping,int64_t local_nodes,
    const at::Tensor& work,const at::Tensor& error,
    const ReverseGatherInput& values,const ReverseGatherInput& gradients);
} // namespace tide::device_online
