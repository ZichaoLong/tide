#pragma once
#include "tide/placement.h"
#include "tide/read.h"
#include "tide/region.h"

namespace tide::placement_detail {
bool builtin_read(const ReadKernel&, const Node&);
bool builtin_region(const RegionKernel&, const Region&);
bool builtin_lh(const RegionKernel&);
std::shared_ptr<const ReadKernel> read_kernel(const Node&, const NodeWeights&, const ResolvedPlacement&);
std::shared_ptr<const RegionKernel> region_kernel(const Region&, const RegionWeights&, const ResolvedPlacement&);
}  // namespace tide::placement_detail
