#pragma once
#include "CortexNet.h"
#include "tide/types.h"
#include <array>
#include <stdexcept>

namespace lh_iocortex {
using tide::Index;
using AccumulateLocal::PtrBatchPtrBaseHidden;
using AccumulateLocal::VPtrBatchPtrBaseHidden;
using Rows = std::map<tide::Owner, at::Tensor>;  // (sample, local node)
using Cache = std::map<std::string, at::Tensor>;
inline void require(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error(message);
}
void close(const at::Tensor&, const at::Tensor&, const std::string&);
struct Policy {
  std::string pool;
  bool clear, lead;
  int original_mode;
  Index layers() const { return clear ? 3 : 2; }
  bool bias() const { return lead; }
};
struct Tick {
  Rows proposals;
  std::map<tide::Owner, Cache> caches;
  VPtrBatchSignals selected;
};
struct ObservedSelector final : BaseSelector {
  NaiveSelector original;
  const VPtrBatchPtrBaseHidden& hidden;
  at::TensorOptions opts;
  std::vector<Tick> ticks;
  ObservedSelector(Index batch, const std::shared_ptr<GraphConfig>&, const Policy&,
                   const VPtrBatchPtrBaseHidden&, at::TensorOptions);
  VPtrBatchSignals select(const VPtrBatchSignals&) override;
};
struct Fixture {
  static constexpr Index width = 4, batch = 4, vocab = 7;
  Policy policy;
  at::TensorOptions opts;
  std::shared_ptr<GraphConfig> cfg;
  std::shared_ptr<IOCortexNet> original;
  tide::Graph body, readout;
  tide::Model model, read_model;
  std::array<BaseCortexNet*, 4> blocks;
  std::array<Index, 4> edge_offsets;
  Index n;
  Fixture(at::TensorOptions, Policy);
};
Cache original_cache(const PtrBatchPtrBaseHidden&, Index, at::TensorOptions);
tide::State empty_state(const tide::Graph&, const tide::Model&, Index);
void compare_cache(const tide::NodeWeights&, const tide::State&, Index cut,
                   const Cache&, const std::string& context);
void compare_states(const Fixture&, const tide::Continuation&,
                    const VPtrBatchPtrBaseHidden&, const VPtrBatchPtrBaseHidden&);
void compare_counts(const Fixture&, const tide::Continuation&,
                    const ObservedSelector&, const ObservedSelector&);
void compare_trace(const Fixture&, const tide::Result&, const ObservedSelector&,
                   const ObservedSelector&, Index start);
void compare_pending(const Fixture&, const tide::Continuation&,
                     const VPtrBatchSignals&, const VPtrBatchSignals&);
void compare_continuation(const tide::Continuation&, const tide::Continuation&);
void write_fixture(const std::string&, const Fixture&, const std::vector<tide::External>&,
                   const ObservedSelector&, const ObservedSelector&,
                   const VPtrBatchPtrBaseHidden&, const VPtrBatchPtrBaseHidden&,
                   const PtrBatchPtrBaseHidden&, const std::vector<at::Tensor>&);
}  // namespace lh_iocortex
