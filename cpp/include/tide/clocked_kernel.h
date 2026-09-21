#pragma once
#include "tide/kernel.h"

namespace tide {
State local_state(const StateClock&, const State&);
State global_state(const StateClock&, const State&);
StateClock kernel_clock(const std::shared_ptr<const StateKernel>&);
std::shared_ptr<const StateKernel> with_state_clock(std::shared_ptr<const StateKernel>, const StateClock&);
}  // namespace tide
