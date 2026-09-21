#pragma once
#include "tide/frontier.h"

namespace tide {
class Specialized {
 public:
  Specialized(Graph, Model, Options, std::string topology);
  Result run(const Continuation&, const std::vector<External>&, Index stop, Index seal);
 private:
  Graph graph_;
  Model model_;
  Options options_;
  std::string topology_;
  NodePool pool_;
};
}  // namespace tide
