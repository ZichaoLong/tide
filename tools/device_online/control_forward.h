#pragma once
#include "content_profile.h"
namespace tide::device_online {
ActionBatch append_control_forward(DeviceProgram&,const ContentProfile&,const ActionBatch&,
    const at::Tensor& content,const at::Tensor& controls,const at::Tensor& error);
}
