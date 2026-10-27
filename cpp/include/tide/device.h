#pragma once
#include "tide/types.h"

namespace tide {
// The build declares its accelerator family; CPU remains the reference device.
bool supported_payload(const Tensor& value);
// Model/session kernels and native checkpoint v2 additionally accept FP16.
// FP16 optimizer updates require the separately explicit FP32 master policy.
bool supported_kernel_payload(const Tensor& value);
const char* execution_backend() noexcept;
}  // namespace tide
