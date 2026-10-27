#pragma once
#include "tide/weighted.h"
#include "tide/placement.h"
namespace tide::weighted {
bool is_describe(const ReadKernel&);
std::optional<RegionalConfig> sending_config(const RegionKernel&);
Tensor placement_copy(const Tensor&,at::Device,at::ScalarType);
void placement_tensors(const std::vector<Tensor*>&,at::Device,at::ScalarType);
std::shared_ptr<const ReadKernel> place_read(const NodeWeights&,const ResolvedPlacement&);
std::shared_ptr<const RegionKernel> place_region(const RegionWeights&,const ResolvedPlacement&);
Selection region_step(const RegionWeights&,const RegionInput&,const RegionalConfig&,
                      std::optional<at::Device> selection=std::nullopt);
}
