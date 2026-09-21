#pragma once
#include "lh_iocortex_fixture.h"

namespace lh_iocortex {
// Bounded fixture adapter, not a converter for arbitrary custom graph programs.
struct SingleGraph {
  tide::Graph graph;
  tide::Model model;
  tide::StateClock body_clock, read_clock;
  Index body_nodes, body_regions, body_edges, layers;
  std::vector<Index> edge_origin;
  std::vector<std::vector<Index>> output_origin;
  explicit SingleGraph(const Fixture&);
  tide::Atom body_atom(tide::Atom) const;
  tide::Result body_view(const tide::Result&, const tide::Graph&) const;
  tide::Continuation read_view(const tide::Continuation&, const tide::Graph&) const;
  std::vector<tide::Output> buffer(const tide::Continuation&) const;
};
void compare_result(const tide::Result&, const tide::Result&);
Index check_single_graph(const Fixture&, int schedule, const std::vector<tide::External>&,
                         const ObservedSelector&, const ObservedSelector&,
                         const std::vector<at::Tensor>& logits = {});
}  // namespace lh_iocortex
