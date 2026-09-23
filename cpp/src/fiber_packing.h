#pragma once
#include "tide/kernel.h"

namespace tide {
PackedStates fiber_attention_packed(const NodeWeights&, Index heads, const std::string& pool, const std::vector<State>&,
                                   const PackedSequence&, const std::string& pooling, const std::string& cache,
                                   const std::string& layout);
PackedStates fiber_attention_single(const NodeWeights&, Index heads, const std::string& pool, const std::vector<State>&,
                                   const PackedSequence&, const std::string& pooling, const std::string& cache,
                                   const std::string& layout);
}  // namespace tide
