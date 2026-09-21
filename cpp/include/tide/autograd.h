#pragma once
#include "tide/types.h"

namespace tide {
Tensor semantic_value(const Tensor& packed, const Tensor& reference);
State semantic_state(const State& packed, const State& reference);
}  // namespace tide
