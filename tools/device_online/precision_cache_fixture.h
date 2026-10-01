#pragma once
#include "tide/kernel.h"

namespace tide::device_online::test {
// Test-only independent CPU scalar state programs; nullptr for other memories.
std::shared_ptr<const StateKernel> half_cache_kernel(const Node&,Index inputs);
} // namespace tide::device_online::test
