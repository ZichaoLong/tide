#pragma once
#include "tide/pool.h"
#include "tide/types.h"

namespace tide {
class Streaming {
 public:
  Streaming(Graph graph, Model model, Options options);
  Result run(const Continuation&, const std::vector<External>&, Index stop, Index seal);
  const Graph& graph() const { return graph_; }
 private:
  Graph graph_;
  Model model_;
  Options options_;
  NodePool pool_;
};
}  // namespace tide
