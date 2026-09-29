#pragma once
#include "tide/types.h"

namespace tide {
// The build declares its accelerator family; CPU remains the reference device.
bool supported_payload(const Tensor& value);
// Model/session compute kernels additionally accept FP16. The standalone
// NamedOptimizer and owner-checkpoint payload contract remains FP32/FP64.
bool supported_kernel_payload(const Tensor& value);
const char* execution_backend() noexcept;
}  // namespace tide
