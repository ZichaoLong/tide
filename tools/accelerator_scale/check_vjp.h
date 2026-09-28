#pragma once
#include "placement.h"

namespace accelerator_scale {
void check_vjp(const Tensor& expected, const Tensor& actual,
               const std::vector<Tensor>& expected_leaves,
               const std::vector<Tensor>& actual_leaves,
               size_t root, bool zero, bool conditioned);
}
