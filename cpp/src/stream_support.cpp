#include "stream_support.h"
#include "tide/kernel.h"
#include "tide/region.h"
#include <algorithm>

namespace tide {
std::vector<Event> compact_stream_events(const Graph& g, const Model& m, const Continuation& q,
                                        std::vector<Atom>& arrived, NodeEvents& by_node, RegionEvents& by_region) {
  std::sort(arrived.begin(), arrived.end(), [](const auto& a, const auto& b) { return a.key() < b.key(); });
  size_t count = 0;
  for (size_t i = 0; i < arrived.size(); ++i)
    if (!i || std::tie(arrived[i].batch, arrived[i].node) != std::tie(arrived[i-1].batch, arrived[i-1].node)) ++count;
  std::vector<Event> events; events.reserve(count);
  for (size_t begin = 0, end; begin < arrived.size(); begin = end) {
    const auto& first = arrived[begin];
    const Owner owner{first.batch, first.node};
    for (end = begin+1; end < arrived.size() && Owner{arrived[end].batch, arrived[end].node} == owner; ++end) {}
    Event e; e.batch = first.batch; e.node = first.node; e.time = first.time;
    e.fiber.reserve(end-begin);
    for (size_t i = begin; i < end; ++i) e.fiber.push_back(std::move(arrived[i]));
    const auto old = q.states.find(owner);
    const auto& w = m.nodes[e.node];
    e.old = old == q.states.end() ? w.kernel->initial(w) : old->second;
    by_node[e.node].push_back(events.size());
    by_region[{e.batch, g.nodes[e.node].region}].push_back(events.size());
    events.push_back(std::move(e));
  }
  return events;
}

void parallel_stream_regions(const Graph& g, const Model& m, Continuation& q, std::vector<Event>& events,
                             const RegionEvents& regions, NodePool& pool, Index workers, bool trace) {
  struct Task { Owner owner; const std::vector<size_t>* ids; const History* old; Selection result; };
  std::vector<Task> tasks; tasks.reserve(regions.size());
  // Stable map entries may be read by workers; no history map mutation until join.
  for (const auto& [owner, ids] : regions) {
    const auto it = q.history.find(owner);
    tasks.push_back({owner, &ids, it == q.history.end() ? nullptr : &it->second, {}});
  }
  const auto count = std::min<size_t>(workers, tasks.size());
  std::vector<std::function<void()>> jobs;
  for (size_t j = 0; j < count; ++j) jobs.push_back([&, j] {
    for (size_t i = tasks.size()*j/count; i < tasks.size()*(j+1)/count; ++i) {
      auto& t = tasks[i]; t.result = evaluate_selection(g, m, t.old, events, *t.ids);
    }
  });
  pool.run(std::move(jobs));
  // Deterministic publication. Completion order never resolves a selection tie.
  for (auto& task : tasks) {
    commit_selection(q, task.owner, std::move(task.result), events, *task.ids, trace);
    const auto observe_all = g.regions[task.owner.second].observe_all;
    for (auto i : *task.ids) {
      auto& e = events[i]; e.comparison_state = observe_all || e.active ? e.proposed_state : e.old;
      e.comparison = e.comparison_state.value;
    }
  }
}

void release_stream_events(std::vector<Event>& events, NodePool& pool, Index workers) {
  const auto count = std::min<size_t>(workers, events.size());
  std::vector<std::function<void()>> jobs;
  for (size_t j = 0; j < count; ++j) jobs.push_back([&, j] {
    for (size_t i = events.size()*j/count; i < events.size()*(j+1)/count; ++i) events[i] = Event{};
  });
  pool.run(std::move(jobs));
}
} // namespace tide
