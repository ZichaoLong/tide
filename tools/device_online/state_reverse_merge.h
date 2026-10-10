#pragma once
#include "state_owner_reverse.h"
namespace tide::device_online {
void append_state_reverse_error(DeviceProgram&,const at::Tensor& source,const at::Tensor& target);
void append_state_reverse_merge(DeviceProgram&,const StateReverseStage&,const StateOwnerVjp&,
    const at::Tensor& ids,const StateVjp& output,const at::Tensor& local_error,const at::Tensor& error);
void append_state_reverse_sources(DeviceProgram&,const StateReversePacket&,const StateOwnerVjp&,
    const at::Tensor& messages,const at::Tensor& connected,const at::Tensor& partials,const at::Tensor& error);
} // namespace tide::device_online
