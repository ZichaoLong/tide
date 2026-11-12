#pragma once
#include "weighted_column_modules.h"
namespace tide::weighted::columns {
at::ScalarType read_dtype(const ResolvedPlacement&,at::ScalarType);
TensorFields read_placed(const NodeWeights&,const NodeBinding&,const LocalWindow&,const std::string&,
                        const LocalState&,const ResolvedPlacement&);
struct PlacedReadRequest {LocalReadRequest query;ResolvedPlacement placement;};
std::vector<TensorFields> read_placed_many(const std::vector<PlacedReadRequest>&,Index max_workspace_bytes);
// A call owns each converted parameter view across actual candidate-count groups.
// The original and these views must outlive all evaluate calls; saved autograd
// edges retain their own inputs. No view is cached across optimizer updates.
class RegionParameterViews {
  const RegionWeights& original_;
  at::Device control_;
  std::map<at::ScalarType,RegionWeights> owned_;
  std::map<at::ScalarType,const RegionWeights*> views_;
 public:
  RegionParameterViews(const RegionWeights& weights,at::Device control):original_(weights),control_(control){}
  const RegionWeights& parameters(at::ScalarType);
  RegionColumns evaluate(const TensorFields&,const RegionalConfig&,const Tensor&,at::Device selection);
};
RegionColumns region_placed(const RegionWeights&,const TensorFields&,const RegionalConfig&,const Tensor&,
                            const ResolvedPlacement&);
Tensor control_payload(const Tensor&,const Tensor& like);
}
