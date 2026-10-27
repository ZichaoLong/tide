#pragma once
#include "tide/weighted.h"
namespace tide::weighted::batch {
std::vector<Selection> regions(const RegionWeights&,const std::vector<RegionInput>&,const RegionalConfig&,
                               std::vector<std::string>* dependencies=nullptr,
                               std::optional<at::Device> selection=std::nullopt);
std::vector<Selection> region_vjp(const RegionWeights&,const std::vector<RegionInput>&,const RegionalConfig&,
                                std::optional<at::Device> selection=std::nullopt);
}
