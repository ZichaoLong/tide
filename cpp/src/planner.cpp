#include "tide/frontier.h"
#include <limits>
#include <stdexcept>

namespace tide {
std::vector<Frame> plan_frontier(const Graph& g, const std::vector<Atom>& atoms, Index start, Index stop, Index limit) {
  const auto order = g.topological_order();
  std::vector<std::set<Owner>> domains(g.nodes.size());  // batch, time
  Index total = 0;
  auto add = [&](Index v, Index b, Index t) {
    if (t >= start && t < stop && domains[v].insert({b, t}).second && ++total > limit)
      throw std::invalid_argument("frontier potential-event limit exceeded; use streaming or a smaller window");
  };
  for (const auto& a : atoms) add(a.node, a.batch, a.time);
  for (auto v : order) {
    for (auto [batch, time] : domains[v]) {
      for (auto j = g.csr.offsets[v]; j < g.csr.offsets[v + 1]; ++j) {
        const auto& edge = g.edges[g.csr.edges[j]];
        if (time <= std::numeric_limits<Index>::max() - edge.delay) add(edge.target, batch, time + edge.delay);
      }
    }
  }
  std::map<Coordinate, std::set<Index>> slots;  // batch, region, time
  for (Index v = 0; v < static_cast<Index>(g.nodes.size()); ++v)
    for (auto [batch, time] : domains[v]) slots[{batch, g.nodes[v].region, time}].insert(v);
  std::vector<Frame> frames;
  std::map<Coordinate, Index> ids;
  for (const auto& [key, nodes] : slots) {
    auto [b, r, t] = key;
    ids[key] = frames.size(); frames.push_back({b, r, t, nodes, {}});
  }
  for (const auto& e : g.edges) {
    for (auto [b, t] : domains[e.source]) {
      if (t > std::numeric_limits<Index>::max() - e.delay) continue;
      auto target = ids.find({b, g.nodes[e.target].region, t + e.delay});
      if (target != ids.end()) frames[target->second].dependencies.insert(ids.at({b, g.nodes[e.source].region, t}));
    }
  }
  return frames;
}
}  // namespace tide
