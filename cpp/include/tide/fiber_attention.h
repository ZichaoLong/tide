#pragma once
#include "tide/kernel.h"

namespace tide {
bool is_fiber_attention_profile(const std::string&);
// Immutable execution policy; not graph identity or checkpoint state.
std::shared_ptr<const StateKernel> make_fiber_attention_kernel(const Node&, Index input_slots = 0,
                                                            const std::string& packing = "exact");
// Select built-in fiber kernels before general model configuration. Custom or
// already configured fiber programs must be chosen explicitly by their owner.
void configure_fiber_attention(const Graph&, Model&, const std::string& packing);
Tensor decode_fiber_bias(const NodeWeights&, const State&, Index cut, std::optional<StateClock> = std::nullopt);
}  // namespace tide
