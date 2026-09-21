#pragma once
#include "tide/kernel.h"

namespace tide {
PackedStates fiber_attention_packed(const NodeWeights&, Index heads, const std::vector<State>&,
                                   const PackedSequence&);
}  // namespace tide
