#pragma once
#include "tide/types.h"

namespace tide {
// Independent vector inputs, shared [out,in] weight. Unlike stack/linear/unbind,
// an unused output returns undefined input gradients, including through upstream
// graphs. A used zero cotangent remains connected. CPU FP32/FP64, first-order VJP.
std::vector<Tensor> isolated_linear(const std::vector<Tensor>& rows, const Tensor& weight);
}  // namespace tide
