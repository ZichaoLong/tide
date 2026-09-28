#include "placement.h"
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace accelerator_scale {
Partition partition(const Graph& graph, const std::vector<int64_t>& bytes,
                    Index count, const std::string& policy) {
  if (count < 1 || bytes.size() != graph.nodes.size() || bytes.empty()
      || (policy != "memory" && policy != "locality"))
    throw std::invalid_argument("invalid node partition request");
  const Index n = bytes.size();
  std::vector<Index> order(n), assigned(n), members(count);
  std::vector<int64_t> loads(count);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](Index a, Index b) { return bytes[a] > bytes[b]; });
  for (auto node : order) {
    if (bytes[node] < 0) throw std::invalid_argument("negative node size");
    auto slot = std::min_element(loads.begin(), loads.end())-loads.begin();
    assigned[node] = slot; loads[slot] += bytes[node]; ++members[slot];
  }
  // Both policies start with exactly the same largest-first memory assignment.
  // Refinement has an explicit cap: 110% of mean plus one indivisible node,
  // or the initial maximum if larger. Embedding/head are placed afterwards.
  const auto total = std::accumulate(bytes.begin(), bytes.end(), int64_t(0));
  const auto limit = std::max(*std::max_element(loads.begin(), loads.end()),
                            (total*11+count*10-1)/(count*10)+bytes[order[0]]);
  if (policy == "memory" || count == 1) return {assigned, limit};
  std::vector<std::vector<Index>> weight(n, std::vector<Index>(n));
  for (const auto& e : graph.edges) if (e.source != e.target) {
    // Every directed physical edge contributes separately, including parallel
    // edges. Direction is immaterial to this symmetric transfer-cost proxy.
    ++weight.at(e.source).at(e.target); ++weight.at(e.target).at(e.source);
  }
  for (Index iteration = 0; iteration < 2*n; ++iteration) {
    std::vector<std::vector<Index>> affinity(n, std::vector<Index>(count));
    for (Index a = 0; a < n; ++a)
      for (Index b = 0; b < n; ++b) affinity[a][assigned[b]] += weight[a][b];
    Index gain = 0, a_best = -1, b_best = -1, destination = -1;
    for (Index a = 0; a < n; ++a) for (Index slot = 0; slot < count; ++slot) {
      if (slot == assigned[a] || members[assigned[a]] <= 1 || loads[slot]+bytes[a] > limit) continue;
      const auto delta = affinity[a][slot]-affinity[a][assigned[a]];
      if (delta > gain) { gain = delta; a_best = a; b_best = -1; destination = slot; }
    }
    // Balanced swaps escape the common case where equal-sized regional nodes
    // cannot move singly. Accept only strict cut reductions; ties are stable.
    for (Index a = 0; a < n; ++a) for (Index b = a+1; b < n; ++b) {
      const auto x = assigned[a], y = assigned[b];
      if (x == y || loads[x]-bytes[a]+bytes[b] > limit || loads[y]-bytes[b]+bytes[a] > limit) continue;
      const auto delta = affinity[a][y]-affinity[a][x]+affinity[b][x]-affinity[b][y]-2*weight[a][b];
      if (delta > gain) { gain = delta; a_best = a; b_best = b; destination = y; }
    }
    if (gain == 0) break;
    const auto old = assigned[a_best];
    loads[old] -= bytes[a_best]; loads[destination] += bytes[a_best];
    --members[old]; ++members[destination]; assigned[a_best] = destination;
    if (b_best >= 0) {
      loads[destination] -= bytes[b_best]; loads[old] += bytes[b_best];
      --members[destination]; ++members[old]; assigned[b_best] = old;
    }
  }
  return {assigned, limit};
}
}  // namespace accelerator_scale
