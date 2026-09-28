#pragma once
#include "placement.h"
#include <tide/pool.h>
#include <c10/core/Stream.h>

namespace accelerator_scale {
// Benchmark-owned sealed-window executor for this exact historical workload.
// It reuses public local programs, with an independent CPU scalar oracle.
// No checkpoint import or claim of general heterogeneous core support.
class Resident {
 public:
  Resident(Graph, Model, Options, Index batch, Placement);
  AdvanceResult advance(const std::vector<External>&, Index stop, Index seal);
  Continuation snapshot() const;
  const std::string& identity() const { return graph_.identity; }
 private:
  using Groups = std::map<Index, std::vector<size_t>>;
  Graph graph_;
  Model model_, selection_model_;
  Options options_;
  Placement placement_;
  NodePool pool_;
  std::vector<c10::Stream> streams_;
  std::vector<NodeWeights> host_read_;
  Continuation state_;
  EventQueue queue_;
  bool failed_ = false;
  void phase(const Groups&, const std::function<void(Index,const std::vector<size_t>&)>&);
  void read(std::vector<Event>&, const std::vector<size_t>&);
};
class Execution {
 public:
  Execution(Graph, Model, Options, Index batch, const Placement&);
  AdvanceResult advance(const std::vector<External>&, Index stop, Index seal);
  Continuation snapshot() const;
  const std::string& identity() const;
 private:
  std::unique_ptr<Streaming> host_;
  std::unique_ptr<StreamingCursor> cursor_;
  std::unique_ptr<Resident> resident_;
};
}  // namespace accelerator_scale
