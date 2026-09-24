#pragma once
#include "tide/full.h"

namespace tide {
std::vector<Tensor> full_fresh_rows(const NodeWeights&, const std::vector<FullInput>&, bool identity);
}  // namespace tide
