#pragma once
#include "tide/frontier.h"

namespace tide {
struct ComponentPlan;
// Host-owned online positive-delay scheduler. Tensor placement is independent
// of scheduling residence. It does not precompute numerical routes or enumerate
// the complete potential-event domain of a window.
class Greedy {
 public:
  Greedy(Graph graph, Model model, Options options);
  Result run(const Continuation&, const std::vector<External>&, Index stop, Index seal);
 private:
  Result run_window(const Continuation&,const std::vector<External>&,Index stop,Index seal);
  Graph graph_;
  Model model_;
  Options options_;
  NodePool pool_;
  std::shared_ptr<const ComponentPlan> components_;
};
}  // namespace tide
