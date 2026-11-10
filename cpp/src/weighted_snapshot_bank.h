#pragma once
// KV §2–4: call-local storage sharing, independent snapshot gradient roots.
#include "tide/types.h"
namespace tide::weighted::query {
// Undefined means no reusable contiguous overlap; callers use ordinary packing.
Tensor bank_gather(const std::vector<Tensor>&);
Tensor bank_project(const std::vector<Tensor>&,const Tensor& weight,const Tensor& visible);
}
