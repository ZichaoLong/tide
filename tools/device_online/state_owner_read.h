#pragma once
#include "state_owner_tape.h"
#include "state_reverse_stage.h"
namespace tide::device_online {
// Adds the local Read derivative to stage cotangents; returns local parameter
// partials. Complete-region softmax has already run on the coordinator device.
ControlVjp append_state_owner_read(DeviceProgram&,const StateOwnerTape&,const StateReverseStage&,
    const at::Tensor& error,int64_t tensor_budget);
} // namespace tide::device_online
