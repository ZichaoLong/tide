#include "tide/types.h"
#include "tide/ports.h"
#include <sstream>
#include <stdexcept>

namespace tide {
std::vector<Index> Graph::topological_order() const {
  std::vector<Index> degree(nodes.size(), 0), ready, order;
  for (const auto& e : edges) ++degree[e.target];
  for (Index v = 0; v < static_cast<Index>(nodes.size()); ++v) if (!degree[v]) ready.push_back(v);
  while (!ready.empty()) {
    auto v = ready.back(); ready.pop_back(); order.push_back(v);
    for (auto j = csr.offsets[v]; j < csr.offsets[v + 1]; ++j)
      if (!--degree[edges[csr.edges[j]].target]) ready.push_back(edges[csr.edges[j]].target);
  }
  if (order.size() != nodes.size()) throw std::invalid_argument("TimedDAG requires an acyclic node graph");
  return order;
}
void Graph::compile() {
  const auto n = static_cast<Index>(nodes.size());
  auto fail = [](const char* s) { throw std::invalid_argument(s); };
  if (!n || regions.empty()) fail("empty node/region set");
  std::vector<Index> members(regions.size(), 0);
  for (const auto& node : nodes) {
    if (node.region < 0 || node.region >= static_cast<Index>(regions.size())) fail("invalid region owner");
    if (node.kv_heads < 1 || node.query_heads < node.kv_heads || node.query_heads % node.kv_heads || node.window < 0)
      fail("invalid attention heads/window");
    ++members[node.region];
  }
  for (size_t r = 0; r < regions.size(); ++r)
    if (regions[r].budget < 1 || regions[r].budget > members[r]) fail("invalid region budget");
  auto valid = [n](Index v) { return v >= 0 && v < n; };
  for (const auto& e : edges)
    if (!valid(e.source) || !valid(e.target) || e.delay <= 0) fail("invalid positive-delay edge");
  for (auto v : inputs) if (!valid(v)) fail("invalid input owner");
  for (auto v : outputs) if (!valid(v)) fail("invalid output owner");
  auto index = [n](const std::vector<Index>& owners) {
    Adjacency result;
    result.offsets.assign(n + 1, 0);
    for (auto v : owners) ++result.offsets[v + 1];
    for (Index v = 0; v < n; ++v) result.offsets[v + 1] += result.offsets[v];
    auto cursor = result.offsets;
    result.edges.resize(owners.size());
    for (Index e = 0; e < static_cast<Index>(owners.size()); ++e) result.edges[cursor[owners[e]]++] = e;
    return result;
  };
  std::vector<Index> source, target;
  for (const auto& e : edges) { source.push_back(e.source); target.push_back(e.target); }
  csr = index(source); csc = index(target); output_index = index(outputs);
  compile_ports(*this);
  // Collision-free canonical structural identity, independent of object addresses.
  std::ostringstream out;
  out << "tide-graph-v5;n=" << n << ';';
  for (const auto& v : nodes) out << v.region << ',' << v.clear << ',' << v.identity << ','
                                << v.memory.size() << ':' << v.memory << ',' << v.full.size() << ':' << v.full << ','
                                << v.query_heads << ',' << v.kv_heads << ',' << v.window << ';';
  out << "r;";
  for (const auto& r : regions) out << r.budget << ',' << r.observe_all << ',' << r.count_priority << ';';
  out << "e;";
  for (const auto& e : edges) out << e.source << ',' << e.target << ',' << e.delay << ';';
  out << "i;"; for (auto v : inputs) out << v << ',';
  out << "o;"; for (auto v : outputs) out << v << ',';
  out << "ls;"; for (auto v : layout->edge_source) out << v << ',';
  out << "lt;"; for (auto v : layout->edge_target) out << v << ',';
  out << "li;"; for (auto v : layout->input) out << v << ',';
  out << "lo;"; for (auto v : layout->output) out << v << ',';
  identity = out.str();
}
}  // namespace tide
