#pragma once
#include "tide/types.h"

namespace tide {
bool is_lh_full(const std::string&);
void validate_lh_full(const NodeWeights&);
Tensor lh_full_fresh(const NodeWeights&, const Tensor& comparison);
}  // namespace tide
