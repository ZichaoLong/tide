#pragma once
#include "weighted_regional_packing.h"
#include "tide/placement.h"
namespace tide::weighted::columns {
// Same cardinality need not mean the same candidates. This boundary preserves
// whole-node descriptor roots with explicit row/branch support in its VJP.
std::map<Index,Tensor> region_groups(const RegionWeights&,const RegionalConfig&,
    const RegionalPacking&,const RegionalFields&,const ResolvedPlacement&);
}
