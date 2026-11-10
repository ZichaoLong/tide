#pragma once
#include "tide/types.h"
namespace tide::weighted::query {
// Undefined means no bounded reusable layout. Independent snapshot roots survive.
Tensor bank_attention(const Tensor& queries,const std::vector<Tensor>& keys,
    const std::vector<Tensor>& values,const Tensor& weights);
}
