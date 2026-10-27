#pragma once
// Pinned 0770593 architecture/KV/allocation. See docs/weighted-message.md.
#include "tide/kernel.h"
#include "tide/read.h"
#include "tide/region.h"
#include "tide/source_class.h"
#include <array>

namespace tide::weighted {
struct KVConfig {
  Index width=4, summary_width=2, adapter_rank=2, heads=1, ffn_width=8, window=0;
  std::string memory_read="sum";
  double beta_short=.1, epsilon=1e-5, alpha_att=1, alpha_ff=1;
  double position_origin=0, position_scale=1;
  std::vector<std::vector<double>> full_frequencies;
  std::vector<double> read_frequencies, distance_slopes;
  void validate() const;
};
struct RegionalConfig {
  Index hidden_width=4, region_control_width=2, k=1;
  double gamma_max=1, allocation_temperature=1, total_fraction=.5;
  std::string region_control_mode="context";
  void validate() const;
};
struct Allocation {
  Tensor controls, multiplier, free, residual;
  std::string constraint;
};
Allocation allocate_quad(const Tensor& scores, const Tensor& caps, const Tensor& lower,
                         const Tensor& upper, const Tensor& temperature);
// Caller supplies named finite parameters in NodeWeights::extra. This function
// installs immutable C++ kernels; no Python callbacks enter graph execution.
void configure_node(NodeWeights&, const KVConfig&, const RegionalConfig&, const SourceClasses&, double alpha_agg);
void configure_region(RegionWeights&, const RegionalConfig&);
void validate_transport(const Graph&, const Model&);
std::array<Tensor,4> memory(const NodeWeights&, const State&, Index time, const Tensor& u, const KVConfig&);
Tensor current(const NodeWeights&, const Tensor& u, const KVConfig&);
Tensor content(const NodeWeights&, const State&, Index time, const Tensor& u, const KVConfig&);
Tensor normalize(const Tensor&, double epsilon);
}  // namespace tide::weighted
