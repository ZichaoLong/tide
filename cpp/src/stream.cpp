#include "tide/cursor.h"
#include "tide/ops.h"
#include "tide/kernel.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>

namespace tide {
Streaming::Streaming(Graph graph, Model model, Options options)
    : graph_(std::move(graph)), model_(std::move(model)), options_(std::move(options)), pool_(options_.workers) {
  graph_.compile();
  configure_model(graph_, model_);
  validate_model(graph_, model_);
  if (options_.mode != "hard" && options_.mode != "softp" && options_.mode != "hst")
    throw std::invalid_argument("invalid emit mode");
  if (!std::isfinite(options_.zeta)) throw std::invalid_argument("nonfinite zeta");
}
Result Streaming::run(const Continuation& initial, const std::vector<External>& external, Index stop, Index seal) {
  std::lock_guard<std::mutex> lock(run_mutex_);
  auto q = initial;  // Tensor handles preserve graph connectivity.
  auto inputs = validate_window(graph_, model_, q, external, stop, seal);
  EventQueue queue;
  for (const auto& a : q.pending) queue[a.time].push_back(a);
  for (const auto& a : inputs) queue[a.time].push_back(a);
  q.pending.clear();
  auto result = execute(q, queue, stop);
  export_pending(q, queue);
  result.continuation = std::move(q);
  return result;
}
Result Streaming::execute(Continuation& q, EventQueue& queue, Index stop) {
  Result result;
  auto& stats = result.stats;
  stats = {{"candidate_events", 0}, {"logical_times", 0}, {"visited_edges", 0},
           {"update_calls", 0}, {"full_calls", 0}};
  while (!queue.empty() && queue.begin()->first < stop) {
    const Index time = queue.begin()->first;
    auto arrived = std::move(queue.begin()->second);
    queue.erase(queue.begin());
    ++stats["logical_times"];
    std::map<Owner, std::vector<Atom>> fibers;
    for (const auto& a : arrived) fibers[{a.batch, a.node}].push_back(a);
    std::vector<Event> events;
    std::map<Index, std::vector<size_t>> by_node;
    std::map<Owner, std::vector<size_t>> by_region;
    for (auto& [owner, fiber] : fibers) {
      std::sort(fiber.begin(), fiber.end(), [](const auto& a, const auto& b) { return a.key() < b.key(); });
      Event event;
      event.batch = owner.first; event.node = owner.second; event.time = time;
      event.fiber = std::move(fiber);
      auto old = q.states.find(owner);
      const auto& weights = model_.nodes[event.node];
      event.old = old == q.states.end() ? weights.kernel->initial(weights) : old->second;
      by_node[event.node].push_back(events.size());
      by_region[{event.batch, graph_.nodes[event.node].region}].push_back(events.size());
      events.push_back(std::move(event));
    }
    stats["candidate_events"] += events.size();
    std::vector<std::function<void()>> jobs;
    for (const auto& [node, ids] : by_node) {
      stats["update_calls"] += options_.packed ? 1 : ids.size();
      jobs.push_back([&, node, ids] {
        const auto& w = model_.nodes[node];
        std::vector<State> old;
        std::vector<Tensor> content;
        std::vector<Index> times;
        FiberViews fiber_views;
        for (auto i : ids) {
          auto& e = events[i];
          e.content = aggregate(model_, e.fiber);
          if (options_.packed) {
            old.push_back(e.old); content.push_back(e.content); times.push_back(time); fiber_views.push_back(&e.fiber);
          }
          else {
            e.proposed_state = w.kernel->step(w, e.old, e.content, time, e.fiber);
            e.descriptor = w.kernel->read(w, e.old, e.proposed_state, e.content, time, e.fiber);
          }
        }
        if (options_.packed) {
          auto h = at::stack(content);
          auto states = w.kernel->batch(w, old, h, times, fiber_views);
          auto desc = w.kernel->read_batch(w, old, states, h, times, fiber_views);
          for (size_t k = 0; k < ids.size(); ++k) {
            events[ids[k]].proposed_state = states[k]; events[ids[k]].descriptor = desc[k];
          }
        }
        for (auto i : ids) {
          auto& e = events[i];
          e.proposal = e.proposed_state.value;
        }
      });
    }
    pool_.run(std::move(jobs));
    for (const auto& [owner, ids] : by_region) {
      const auto& region = graph_.regions[owner.second];
      auto& history = q.history[owner];
      std::vector<std::tuple<Index, double, Index, size_t>> rank;
      std::vector<Tensor> desc;
      for (auto i : ids) {
        const auto& e = events[i];
        const auto score = e.descriptor.item<double>();
        if (!std::isfinite(score)) throw std::invalid_argument("nonfinite selector score");
        auto count = history.find(e.node);
        rank.emplace_back(region.count_priority && count != history.end() ? count->second : 0,
                          -score, e.node, i);
        desc.push_back(e.descriptor);
      }
      std::sort(rank.begin(), rank.end());
      const auto selected = std::min<Index>(region.budget, rank.size());
      for (Index j = 0; j < selected; ++j) {
        auto& e = events[std::get<3>(rank[j])]; e.active = true; ++history[e.node];
      }
      auto controls = at::softmax(at::stack(desc), 0);
      for (size_t j = 0; j < ids.size(); ++j) {
        auto& e = events[ids[j]];
        e.control = controls[j];
        const State& comparison = region.observe_all || e.active ? e.proposed_state : e.old;
        e.comparison = comparison.value;
        e.comparison_state = comparison;
        e.next_state = graph_.nodes[e.node].clear && e.active ? model_.nodes[e.node].kernel->reset(comparison) : comparison;
        e.next = e.next_state.value;
        q.states[{e.batch, e.node}] = e.next_state;
        if (options_.trace) e.history = history;
      }
    }
    jobs.clear();
    for (const auto& [node, all] : by_node) {
      std::vector<size_t> ids;
      for (auto i : all) if (events[i].active) ids.push_back(i);
      if (ids.empty()) continue;
      stats["full_calls"] += options_.packed ? 1 : ids.size();
      jobs.push_back([&, node, ids] {
        if (options_.packed) {
          std::vector<Tensor> cmp, content, p;
          for (auto i : ids) {
            cmp.push_back(events[i].comparison); content.push_back(events[i].content); p.push_back(events[i].control);
          }
          auto values = full(model_.nodes[node], at::stack(cmp), at::stack(content), at::stack(p), options_, graph_.nodes[node].identity);
          for (size_t j = 0; j < ids.size(); ++j) events[ids[j]].full = values[j];
        } else {
          for (auto i : ids) {
            auto& e = events[i];
            e.full = full(model_.nodes[node], e.comparison, e.content, e.control, options_, graph_.nodes[node].identity);
          }
        }
      });
    }
    pool_.run(std::move(jobs));
    for (auto& event : events) {
      if (event.active) {
        const Index node = event.node;
        for (Index j = graph_.csr.offsets[node]; j < graph_.csr.offsets[node + 1]; ++j) {
          const Index id = graph_.csr.edges[j];
          const auto& edge = graph_.edges[id];
          if (time > std::numeric_limits<Index>::max() - edge.delay)
            throw std::overflow_error("logical time overflow");
          Atom a{event.batch, edge.target, time + edge.delay, 1, id, time, event.full * model_.edge_scale[id]};
          queue[a.time].push_back(a);
          if (options_.trace) result.messages.push_back(a);
          ++stats["visited_edges"];
        }
        for (Index j = graph_.output_index.offsets[node]; j < graph_.output_index.offsets[node + 1]; ++j) {
          const auto port = graph_.output_index.edges[j];
          result.outputs.push_back({event.batch, time, port, event.full * model_.output_scale[port]});
        }
      }
      if (options_.trace) result.trace.push_back(std::move(event));
    }
  }
  q.cut = stop;
  return result;
}
}  // namespace tide
