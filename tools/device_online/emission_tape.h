#pragma once
#include <ATen/ATen.h>
namespace tide::device_online {
// Actual unscaled selected slot values and immutable projection parameters.
// Rows preserve physical local-slot identity, including parallel edges/phases.
struct EmissionTape {at::Tensor metadata,values,count,weights,biases;};
struct EmissionVjp {at::Tensor weights,biases,connected,chunks;};
} // namespace tide::device_online
