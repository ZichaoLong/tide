#pragma once
// Pinned 0770593 architecture/KV/allocation. See docs/weighted-message.md.
#include "tide/kernel.h"
#include "tide/read.h"
#include "tide/region.h"
#include "tide/source_class.h"
#include <array>
#include <optional>
#include <variant>

namespace tide::weighted {
// Marker for the local binding's transport invariant, not a scheduler policy.
class MemoryKernel : public StateKernel {
 public:
  std::string batch_autograd_key(const State&,const ContentView& h,Index) const override {
    return h.value[0].item<double>()>0 ? "positive-write" : "retain-raw";
  }
};
struct KVConfig {
  Index width=4, summary_width=2, adapter_rank=2, heads=1, ffn_width=8, window=0;
  std::string memory_read="sum";
  double beta_short=.1, epsilon=1e-5, alpha_att=1, alpha_ff=1;
  double position_origin=0, position_scale=1;
  std::vector<std::vector<double>> full_frequencies;
  std::vector<double> read_frequencies, distance_slopes;
  void validate() const;
};
struct RecurrentConfig {
  std::string memory_family="affine", memory_read="sum";
  Index width=4, summary_width=2, key_width=2, value_width=3, ffn_width=8;
  Index heads=1, groups=1, conv_size=1, conv_stride=1;
  double lambda_short=.1, epsilon=1e-5, alpha_mem=1, alpha_ff=1;
  double qk_epsilon=1e-6, output_epsilon=1e-5;
  std::vector<double> matrix_rates;
  bool basic() const {return memory_family=="affine"||memory_family=="delta";}
  Index matrices() const {return basic()?2:heads;}
  Index conv_width() const {return basic()?0:(memory_family=="gated_delta"?heads*(2*key_width+value_width):heads*value_width+2*groups*key_width);}
  void validate() const;
};
struct RegionalConfig {
  Index hidden_width=4;
  std::optional<Index> region_control_width=2, k=1;
  double gamma_max=1;
  std::optional<double> allocation_temperature=1, total_fraction, ent_alpha, eta, fixed_gain, gain_logit;
  std::string selection_mode;
  std::optional<std::string> region_control_mode="context", region_evidence="cat", regularizer="quad";
  std::optional<std::string> send_gain, allocation_base;
  std::string local_score="cat", region_score="ind";
  std::optional<Index> selection_width;
  std::optional<std::string> region_key, query_weight;
  std::optional<double> alpha_match, alpha_res, alpha_loc, alpha_sel;
  explicit RegionalConfig(std::string mode="regional_total");
  bool regional() const;
  bool total() const;
  bool hard() const;
  bool attention() const;
  Index capacity(Index size) const;
  void validate() const;
};
// Value-only inspection of installed weighted functions. No tensor evaluation,
// parameter cloning, profile-name parsing or inference from parameter shapes.
using MemoryConfiguration=std::variant<KVConfig,RecurrentConfig>;
struct NodeBinding {
  MemoryConfiguration memory;
  RegionalConfig regional;
  SourceClasses sources;
  double aggregate_alpha;
};
NodeBinding node_binding(const NodeWeights&);
RegionalConfig region_binding(const RegionWeights&);
bool same_configuration(const KVConfig&,const KVConfig&);
bool same_configuration(const RecurrentConfig&,const RecurrentConfig&);
bool same_configuration(const RegionalConfig&,const RegionalConfig&);
struct Allocation {
  Tensor controls, multiplier, free, residual;
  std::string constraint;
  std::string backward="quad-exact-map-v1";
};
Allocation allocate_quad(const Tensor& scores, const Tensor& caps, const Tensor& lower,
                         const Tensor& upper, const Tensor& temperature, const Tensor& score_shift={});
Allocation allocate(const Tensor& scores, const Tensor& caps, const Tensor& lower,
                    const Tensor& upper, const Tensor& temperature, const std::string& regularizer,
                    std::optional<double> ent_alpha=std::nullopt, const Tensor& score_shift={});
struct AllocationRow {Tensor scores,caps,lower,upper,temperature,shift;};
std::vector<Allocation> allocate_batch(const std::vector<AllocationRow>&,const std::string& regularizer,
                                      std::optional<double> ent_alpha=std::nullopt);
// Caller supplies named finite parameters in NodeWeights::extra. This function
// installs immutable C++ kernels; no Python callbacks enter graph execution.
void configure_node(NodeWeights&, const KVConfig&, const RegionalConfig&, const SourceClasses&, double alpha_agg);
void configure_node(NodeWeights&, const RecurrentConfig&, const RegionalConfig&, const SourceClasses&, double alpha_agg);
void configure_region(RegionWeights&, const RegionalConfig&);
void validate_transport(const Graph&, const Model&);
std::array<Tensor,4> memory(const NodeWeights&, const State&, Index time, const Tensor& u, const KVConfig&);
Tensor current(const NodeWeights&, const Tensor& u, const KVConfig&);
Tensor content(const NodeWeights&, const State&, Index time, const Tensor& u, const KVConfig&);
std::array<Tensor,4> memory(const NodeWeights&, const State&, Index time, const Tensor& u, const RecurrentConfig&);
Tensor current(const NodeWeights&, const Tensor& u, const RecurrentConfig&);
Tensor content(const NodeWeights&, const State&, Index time, const Tensor& u, const RecurrentConfig&);
Tensor normalize(const Tensor&, double epsilon);
}  // namespace tide::weighted
