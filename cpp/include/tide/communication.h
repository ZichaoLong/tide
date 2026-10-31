#pragma once
#include "tide/types.h"
#include <string>

namespace tide {
// Each experiment process freezes TIDE_COMMUNICATION_POLICY before executing.
// Callers save this value with autograd edges; it is never inferred from P2P.
std::string communication_policy();
std::map<std::string,Index> communication_statistics(bool reset=false);
Tensor communication_copy(const Tensor&, at::Device, at::ScalarType, const std::string& policy);
inline Tensor communication_copy(const Tensor& value, at::Device device, at::ScalarType dtype) {
  return communication_copy(value, device, dtype, communication_policy());
}
} // namespace tide
