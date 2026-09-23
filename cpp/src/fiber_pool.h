#pragma once
#include "tide/types.h"

namespace tide {
std::string fiber_pool_kind(const std::string& profile);
bool fiber_pool_learned(const std::string& kind);
void validate_fiber_pool(const NodeWeights&, const std::string& kind, Index input_slots);
Tensor fiber_pool_rows(const NodeWeights&, const std::string& kind, const std::vector<Index>& slots,
                       const Tensor& rows);
Tensor fiber_pool_csr(const NodeWeights&, const std::string& kind, const std::vector<Index>& slots,
                      const std::vector<Index>& offsets, const std::vector<Tensor>& event_rows);
}  // namespace tide
