#pragma once
#include "tide/types.h"

namespace tide {
// Undefined cotangents leave rows disconnected; numerical zeros stay connected.
std::vector<Tensor> isolated_read(const std::vector<Tensor>& rows, const Tensor& weight,
                                  bool norm, at::ScalarType dtype, at::Device device);
}
