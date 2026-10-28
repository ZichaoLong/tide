#pragma once
#include <ATen/ATen.h>
#include <vector>
#include <map>
#include <tuple>
#include <cstdint>

namespace tide {
// Materialize one FP32 numerical role from a CPU parameter. Admission precedes
// reshape/copy and charges actual role storage, independently of parameter ties.
at::Tensor copy_parameter_bank(const at::Tensor&,const std::vector<int64_t>& shape,
                              int64_t offset,at::Device,int64_t& available_bytes);
// Private frozen-owner pool. Reuses only identical source TensorImpl, slice and
// device. Returned banks are mutable only through coordinated publication;
// publication must deduplicate identical destinations before launching writes.
class ParameterBankPool {
 public:
  at::Tensor bind(const at::Tensor&,const std::vector<int64_t>& shape,int64_t offset,at::Device,int64_t& available_bytes);
 private:
  struct Source {at::Tensor value;int64_t version;};
  std::map<std::uintptr_t,Source> sources_;
  std::map<std::tuple<std::uintptr_t,int64_t,int64_t,int,int>,at::Tensor> banks_;
};
}
