#pragma once
#include "weighted_column_modules.h"
namespace tide::weighted::columns {
at::ScalarType read_dtype(const ResolvedPlacement&,at::ScalarType);
TensorFields read_placed(const NodeWeights&,const NodeBinding&,const LocalWindow&,const std::string&,
                        const LocalState&,const ResolvedPlacement&);
RegionColumns region_placed(const RegionWeights&,const TensorFields&,const RegionalConfig&,const Tensor&,
                            const ResolvedPlacement&);
Tensor control_payload(const Tensor&,const Tensor& like);
}
