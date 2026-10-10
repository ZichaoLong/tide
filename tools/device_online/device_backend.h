#pragma once
#include <c10/core/Device.h>

#ifndef TIDE_RESIDENT_CUDA
#define TIDE_RESIDENT_CUDA 0
#endif
namespace tide::device_online {
inline constexpr auto resident_device_type = TIDE_RESIDENT_CUDA
    ? c10::DeviceType::CUDA : c10::DeviceType::PrivateUse1;
// Verify the actual backend before constructing a device program. CPU buffers
// can be boundary inputs, but never substitute for resident control/storage.
void validate_kernel_device(c10::Device);
void check_device_launch(int status,const char* operation);
}
