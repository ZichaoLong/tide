#pragma once
#include "tide/kernel.h"

namespace tide {
bool is_fiber_attention_profile(const std::string&);
std::shared_ptr<const StateKernel> make_fiber_attention_kernel(const Node&, Index input_slots = 0);
Tensor decode_fiber_bias(const NodeWeights&, const State&, Index cut);
}  // namespace tide
