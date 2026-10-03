#pragma once
#include "tide/types.h"

namespace tide {
// Region tensors own their history; parameterless regions follow their first
// member. Execution placement can compute Read/control on a different device.
const Tensor& region_reference(const Graph&, const Model&, Index region);
std::vector<at::Device> model_devices(const Model&);
// Construction-time placement: returns new leaves for moved tensors, preserving
// aliases. Create the optimizer after this call. Shared node weights constrain
// node co-location; conflicting maps fail before copying or executing anything.
Model place_payloads(const Graph&, const Model&, const std::vector<at::Device>&);
}  // namespace tide
