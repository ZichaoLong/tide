#pragma once
#include "tide/types.h"

namespace tide {
// The caller provides all selected-port body outputs in the sealed interval
// [readout.cut * layers, stop * layers). No continuation or tensor is mutated.
std::vector<External> token_inputs(const std::vector<Output>&, const Continuation& readout,
                                  Index layers, Index stop, Index body_cut, Index port = 0);
}  // namespace tide
