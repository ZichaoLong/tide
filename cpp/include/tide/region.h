#pragma once
#include "tide/types.h"
#include <set>

namespace tide {
// Static policy and membership are graph-owned; views are borrowed synchronously.
struct RegionLayout {
  const Region& spec;
  c10::ArrayRef<Index> members;
  Index slot(Index node) const;
};
struct Candidate { Index node; Tensor descriptor; };
struct RegionInput {
  const History& history;
  Index time;
  c10::ArrayRef<Candidate> candidates;
  RegionLayout layout;
  at::TensorOptions payload_options;
};
struct Selection {
  std::set<Index> active;
  std::map<Index, Tensor> controls;
  History history;
};
class RegionKernel {
 public:
  virtual ~RegionKernel() = default;
  virtual History initial(const RegionWeights&, const RegionLayout&, const Tensor& reference) const { return {}; }
  virtual Selection step(const RegionWeights&, const RegionInput&) const = 0;
  virtual void validate_weights(const RegionWeights&, const RegionLayout&) const = 0;
  virtual void validate_history(const History&, const RegionLayout&) const = 0;
};
RegionLayout region_layout(const Graph&, Index region);
std::shared_ptr<const RegionKernel> make_region_kernel(const Region&);
void validate_history(const History&, const RegionLayout&, const Tensor& reference, Index time);
Selection evaluate_selection(const Graph&, const Model&, const History*, const std::vector<Event>&,
                             const std::vector<size_t>&);
void commit_selection(Continuation&, Owner, Selection, std::vector<Event>&,
                      const std::vector<size_t>&, bool trace);
void select_events(const Graph&, const Model&, Continuation&, std::vector<Event>&,
                   const std::vector<size_t>&, bool trace);
}  // namespace tide
