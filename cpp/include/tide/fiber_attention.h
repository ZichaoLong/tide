#pragma once
#include "tide/kernel.h"

namespace tide {
std::shared_ptr<const StateKernel> make_fiber_attention_kernel(const Node&);
Tensor decode_fiber_bias(const NodeWeights&, const State&, Index cut);
}  // namespace tide
