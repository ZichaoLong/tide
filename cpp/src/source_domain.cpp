#include "tide/source_domain.h"
#include <set>
#include <stdexcept>

namespace tide {
void compile_source_domain(Graph& g) {
  if (!g.source_domain) g.source_domain = SourceDomain{g.layout->edge_target, g.layout->input};
  const auto& domain = *g.source_domain;
  if (domain.edge_target.size() != g.edges.size() || domain.input.size() != g.inputs.size())
    throw std::invalid_argument("source domain size mismatch");
  std::vector<std::set<Index>> rows(g.nodes.size());
  for (size_t e = 0; e < g.edges.size(); ++e) rows[g.edges[e].target].insert(domain.edge_target[e]);
  for (size_t p = 0; p < g.inputs.size(); ++p) rows[g.inputs[p]].insert(domain.input[p]);
  g.source_counts.clear();
  for (const auto& row : rows) {
    if (!row.empty() && (*row.begin() != 0 || *row.rbegin() != static_cast<Index>(row.size()) - 1))
      throw std::invalid_argument("source domain must be dense nonnegative slots for each node");
    g.source_counts.push_back(row.size());
  }
}
}  // namespace tide
