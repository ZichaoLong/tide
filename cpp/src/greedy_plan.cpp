#include "greedy_plan.h"
#include <algorithm>
#include <functional>
#include <queue>
#include <stdexcept>

namespace tide {
ClosurePlan::ClosurePlan(const Graph& g):ClosurePlan(g,[&]{std::vector<Index> ids;for(size_t i=0;i<g.regions.size();++i)ids.push_back(i);return ids;}()){}
ClosurePlan::ClosurePlan(const Graph& g,const std::vector<Index>& regions):ClosurePlan(g,regions,[&]{std::vector<Index> ids;for(size_t i=0;i<g.edges.size();++i)ids.push_back(i);return ids;}()){}
ClosurePlan::ClosurePlan(const Graph& g,const std::vector<Index>& regions,const std::vector<Index>& wires) : outgoing_(regions.size()) {
  for(size_t i=0;i<regions.size();++i)local_[regions[i]]=i;
  std::map<Owner, Index> edges;
  for (auto id : wires) {
    const auto& edge=g.edges[id];
    Owner pair{g.nodes[edge.source].region, g.nodes[edge.target].region};
    if(!local_.count(pair.first)||!local_.count(pair.second))continue;
    pair={local_.at(pair.first),local_.at(pair.second)};
    auto [it, inserted] = edges.emplace(pair, edge.delay);
    if (!inserted) it->second = std::min(it->second, edge.delay);
  }
  for (auto [pair, delay] : edges) outgoing_[pair.first].push_back({pair.second, delay});
}
std::map<Index, std::vector<Frame>> ClosurePlan::ready(const Graph& g, const Fibers& fibers,
                                                      Index stop, Index limit, Index& relaxed) const {
  if (fibers.size() > size_t(limit)) throw std::invalid_argument("greedy live-fiber capacity exceeded");
  std::map<Coordinate, std::set<Index>> frames;
  std::map<Index, std::map<Index, Index>> seeds;
  for (const auto& [coordinate, atoms] : fibers) {
    auto [batch, node, time] = coordinate;
    if (atoms.empty() || time >= stop) continue;
    const auto region = g.nodes[node].region;
    if(!local_.count(region))continue;
    frames[{batch, region, time}].insert(node);
    auto [it, inserted] = seeds[batch].emplace(local_.at(region), time);
    if (!inserted) it->second = std::min(it->second, time);
  }
  // Saturate at stop without floating conversion or signed integer overflow.
  auto arrival = [stop](Index time, Index delay) {
    return delay >= stop-time ? stop : time+delay;
  };
  std::map<Index, std::vector<Index>> bounds;
  for (const auto& [batch, initial] : seeds) {
    std::vector<Index> earliest(outgoing_.size(), stop);
    std::priority_queue<Owner, std::vector<Owner>, std::greater<Owner>> todo;
    for (auto [region, time] : initial) { earliest[region] = time; todo.push({time, region}); }
    while (!todo.empty()) {
      const auto [time, source] = todo.top(); todo.pop();
      if (time != earliest[source]) continue;
      for (auto [target, delay] : outgoing_[source]) {
        ++relaxed;
        const auto next = arrival(time, delay);
        if (next < earliest[target]) { earliest[target] = next; todo.push({next, target}); }
      }
    }
    auto& safe = bounds[batch]; safe.assign(outgoing_.size(), stop);
    for (size_t source=0; source<outgoing_.size(); ++source)
      for (auto [target, delay] : outgoing_[source])
        safe[target] = std::min(safe[target], arrival(earliest[source], delay));
  }
  std::map<Index, std::vector<Frame>> blocks;
  for (const auto& [coordinate, nodes] : frames) {
    const auto [batch, region, time] = coordinate;
    if (time < bounds.at(batch)[local_.at(region)]) blocks[region].push_back({batch, region, time, nodes, {}});
  }
  for (auto& [region, ready] : blocks)
    std::sort(ready.begin(), ready.end(), [](const auto& a, const auto& b) {
      return std::tie(a.time, a.batch) < std::tie(b.time, b.batch);
    });
  if (!frames.empty() && blocks.empty()) throw std::logic_error("greedy closure failed to expose the earliest event");
  return blocks;
}
}  // namespace tide
