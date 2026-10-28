#pragma once
#include <ATen/ATen.h>
#include <vector>

namespace tide {
// Materialize one FP32 numerical role from a CPU parameter. Admission precedes
// reshape/copy and charges actual role storage, independently of parameter ties.
at::Tensor copy_parameter_bank(const at::Tensor&,const std::vector<int64_t>& shape,
                              int64_t offset,at::Device,int64_t& available_bytes);
}
