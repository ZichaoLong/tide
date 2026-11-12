#pragma once
#include "tide/types.h"
namespace tide::weighted {
// KV §1.2: actual BMM with independent first-order roots and original versions.
std::vector<Tensor> linear_rows(const std::vector<Tensor>&,const std::vector<Tensor>&);
// Whole R×D node operands; no per-row autograd views or parameter outer products.
std::vector<Tensor> linear_blocks(const std::vector<Tensor>&,const std::vector<Tensor>&);
}
