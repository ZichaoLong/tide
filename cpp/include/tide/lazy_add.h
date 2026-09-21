#pragma once
#include "tide/kernel.h"

namespace tide {
std::shared_ptr<const StateKernel> make_add_repeat_kernel();
// Physical hidden after [0,cut); never changes stored clocks or state.
Tensor decode_add_repeat(const NodeWeights&, const State&, Index cut, std::optional<StateClock> = std::nullopt);
}  // namespace tide
