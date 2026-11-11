#pragma once
// 0770593 allocation §2–6/S.2. Complete R×N tensor groups, not event objects.
// R may flatten both sample and logical time. Coordinates are owned by callers.
#include "tide/weighted.h"

namespace tide::weighted {
struct ColumnVersionGuard;
struct AllocationColumns {
  Tensor controls,multiplier,branch,response,free,residual;
  std::array<Tensor,6> inputs; // scores,caps,lower,upper,temperature,optional shift
  std::string regularizer,backward;
  std::optional<double> alpha;
  std::shared_ptr<ColumnVersionGuard> versions;
};
AllocationColumns allocate_columns(const Tensor& scores,const Tensor& caps,const Tensor& lower,
    const Tensor& upper,const Tensor& temperature,const std::string& regularizer="quad",
    std::optional<double> alpha=std::nullopt,const Tensor& shift={});
// Exact-map first-order transpose. row_support is bool R, independent of a
// numerically zero adjoint. None/undefined is returned per whole input column.
std::array<Tensor,6> allocation_columns_vjp(const AllocationColumns&,const Tensor& bar,
                                          const Tensor& row_support={});
struct RegionColumns {
  Tensor controls;
  std::optional<AllocationColumns> allocation;
  std::string mode;
};
RegionColumns region_columns(const RegionWeights&,const TensorFields&,const RegionalConfig&,
    const Tensor& nodes={},std::optional<at::Device> selection=std::nullopt);
std::vector<std::string> region_column_dependencies(const RegionColumns&);
std::vector<bool> region_column_differentiable(const RegionColumns&);
std::vector<std::pair<Tensor,Tensor>> region_column_vjp_sources(
    const RegionColumns&,const Tensor& bar,const Tensor& row_support);
}
