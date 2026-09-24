#pragma once
#include "tide/types.h"

namespace tide {
// Homogeneous source signature, event-major atoms/scales. Returns, per event,
// summary followed by source contributions. Coefficients: [sources] shared or
// [events,sources] caller-owned; empty for sum/mean. First-order, undefined-safe VJP.
std::vector<Tensor> isolated_aggregate(const std::vector<Tensor>& atoms,
    const std::vector<Tensor>& scales, const Tensor& coefficients, Index sources, bool mean);
} // namespace tide
