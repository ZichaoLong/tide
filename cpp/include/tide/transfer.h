#pragma once
#include "tide/types.h"

namespace tide {
constexpr Index transfer_pack_bytes = 8*1024*1024;
// Equal vector rows copied in bounded groups. Unused public outputs preserve
// undefined input cotangents; present numerical zeros remain connected. A row
// larger than the packing budget uses one ordinary, indivisible tensor copy.
std::vector<Tensor> copy_rows(const std::vector<Tensor>& rows, at::Device destination,
                              Index budget = transfer_pack_bytes);
} // namespace tide
