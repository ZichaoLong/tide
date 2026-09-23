#pragma once
#include "tide/pool.h"
#include "tide/types.h"
#include <set>

namespace tide {
using Coordinate = std::tuple<Index, Index, Index>;  // batch, node, time
using Fibers = std::map<Coordinate, std::vector<Atom>>;
struct Frame {
  Index batch, region, time;
  std::set<Index> nodes, dependencies;
};
std::vector<Frame> plan_frontier(const Graph&, const std::vector<Atom>&, Index start, Index stop, Index limit);
std::vector<Event> evaluate_block(const Graph&, const Model&, Continuation&, const std::vector<Frame>&,
                                 Fibers&, const Options&, NodePool&, std::map<std::string, Index>& stats);
class Frontier {
 public:
  Frontier(Graph graph, Model model, Options options);
  Result run(const Continuation&, const std::vector<External>&, Index stop, Index seal);
 private:
  Graph graph_;
  Model model_;
  Options options_;
  NodePool pool_;
};
}  // namespace tide
