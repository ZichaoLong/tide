#pragma once
// 0770593 causal-convolution §1–3: exact finite map at original logical times.
#include "tide/types.h"

namespace tide::weighted {
struct ConvCache {Tensor times, values;};
void validate_cache(const ConvCache&, Index width, Index horizon);
Tensor convolve(const ConvCache&, Index time, const Tensor& current,
                const Tensor& weight, const Tensor& bias, Index stride);
ConvCache conv_write(const ConvCache&, Index time, const Tensor& receive,
                    const Tensor& current, Index size, Index stride);
std::pair<Tensor,Tensor> conv_effective(const ConvCache&, Index time, Index size, Index stride);
}
