#pragma once
#include "placement.h"

namespace accelerator_scale {
void check_vjp(const Tensor& expected, const Tensor& actual,
               const std::vector<Tensor>& expected_leaves,
               const std::vector<Tensor>& actual_leaves,
               size_t root, bool zero, bool conditioned, double rtol=1e-5, double atol=1e-6, bool half=false);
}
