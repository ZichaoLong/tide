#pragma once
#include "tide/types.h"

namespace tide {
// The build declares its accelerator family; CPU remains the reference device.
bool supported_payload(const Tensor& value);
const char* execution_backend() noexcept;
}  // namespace tide
