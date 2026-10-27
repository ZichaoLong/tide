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
struct Candidate {
  Index node;
  Tensor descriptor;
  std::map<std::string, Tensor> fields;
};
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
  virtual bool joint_batch() const { return false; }
  virtual bool batched_autograd() const { return false; }
  virtual bool history_independent() const { return false; }
  virtual std::vector<Selection> batch(const RegionWeights& w,const std::vector<RegionInput>& r) const {
    std::vector<Selection> out;for(const auto& row:r)out.push_back(step(w,row));return out;
  }
  virtual std::vector<Selection> batch_grad(const RegionWeights&,const std::vector<RegionInput>&) const;
  virtual void validate_weights(const RegionWeights&, const RegionLayout&) const = 0;
  virtual void validate_history(const History&, const RegionLayout&) const = 0;
};
RegionLayout region_layout(const Graph&, Index region);
std::shared_ptr<const RegionKernel> make_region_kernel(const Region&);
void validate_history(const History&, const RegionLayout&, const Tensor& reference, Index time);
Selection evaluate_selection(const Graph&, const Model&, const History*, const std::vector<Event>&,
                             const std::vector<size_t>&);
void select_event_batches(const Graph&,const Model&,Continuation&,std::vector<Event>&,
                          const std::map<Owner,std::vector<size_t>>&,bool trace,std::map<std::string,Index>& stats);
void commit_selection(Continuation&, Owner, Selection, std::vector<Event>&,
                      const std::vector<size_t>&, bool trace);
void select_events(const Graph&, const Model&, Continuation&, std::vector<Event>&,
                   const std::vector<size_t>&, bool trace);
}  // namespace tide
