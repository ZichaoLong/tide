#pragma once
#include "weighted_column_modules.h"
namespace tide::weighted::columns {
at::ScalarType read_dtype(const ResolvedPlacement&,at::ScalarType);
TensorFields read_placed(const NodeWeights&,const NodeBinding&,const LocalWindow&,const std::string&,
                        const LocalState&,const ResolvedPlacement&);
struct PlacedReadRequest {LocalReadRequest query;ResolvedPlacement placement;};
std::vector<TensorFields> read_placed_many(const std::vector<PlacedReadRequest>&,Index max_workspace_bytes);
RegionColumns region_placed(const RegionWeights&,const TensorFields&,const RegionalConfig&,const Tensor&,
                            const ResolvedPlacement&);
Tensor control_payload(const Tensor&,const Tensor& like);
}
