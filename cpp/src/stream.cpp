#include "tide/cursor.h"
#include "tide/ops.h"
#include "tide/kernel.h"
#include "tide/autograd.h"
#include "tide/full.h"
#include "tide/aggregate.h"
#include "tide/read.h"
#include "tide/next.h"
#include "tide/region.h"
#include "tide/delivery.h"
#include <ATen/core/grad_mode.h>
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
           {"update_calls", 0}, {"full_calls", 0},
           {"semantic_state_replays", 0}, {"semantic_full_replays", 0}, {"full_scalar_fallback_steps", 0}};
  const bool replay = at::GradMode::is_enabled();
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
      stats["read_calls"] += options_.packed ? 1 : ids.size();
      if (options_.packed && replay) stats["semantic_read_replays"] += ids.size();
      if (options_.packed && !model_.nodes[node].read_kernel->joint_batch()) stats["read_scalar_batch_steps"] += ids.size();
      if (options_.packed && replay) stats["semantic_state_replays"] += ids.size();
      if (options_.packed && !model_.nodes[node].kernel->joint_batch()) stats["state_scalar_batch_steps"] += ids.size();
      stats["aggregate_calls"] += options_.packed ? 1 : ids.size();
      if (options_.packed && replay) stats["semantic_aggregate_replays"] += ids.size();
      if (options_.packed && !model_.nodes[node].aggregate_kernel->joint_batch()) stats["aggregate_scalar_fallback_steps"] += ids.size();
      jobs.push_back([&, node, ids] {
        const auto& w = model_.nodes[node];
        evaluate_aggregate(graph_, model_, events, ids, options_.packed);
        std::vector<State> old;
        std::vector<Tensor> content;
        std::vector<Index> times;
        ContentViews content_views;
        for (auto i : ids) {
          auto& e = events[i];
          if (options_.packed) {
            old.push_back(e.old); content.push_back(e.content); times.push_back(time); content_views.push_back(e.local_content());
          }
          else {
            e.proposed_state = w.kernel->step(w, e.old, e.local_content(), time);
          }
        }
        if (options_.packed) {
          std::vector<State> states;
          {
            at::NoGradGuard guard;
            auto h = at::stack(content);
            states = w.kernel->batch(w, old, h, times, content_views);
            if (states.size() != ids.size()) throw std::invalid_argument("state batch changed event count");
          }
          for (size_t k = 0; k < ids.size(); ++k) {
            auto& e = events[ids[k]];
            e.proposed_state = states[k];
            if (replay) {
              auto reference = w.kernel->step(w, e.old, e.local_content(), time);
              e.proposed_state = semantic_state(e.proposed_state, reference);
            }
          }
        }
        evaluate_read(graph_, model_, events, ids, options_.packed);
        for (auto i : ids) {
          auto& e = events[i];
          e.proposal = e.proposed_state.value;
        }
      });
    }
    pool_.run(std::move(jobs));
    for (const auto& [owner, ids] : by_region) {
      const auto& region = graph_.regions[owner.second];
      select_events(graph_, model_, q, events, ids, options_.trace);
      ++stats["region_steps"];
      for (auto i : ids) {
        auto& e = events[i];
        const State& comparison = region.observe_all || e.active ? e.proposed_state : e.old;
        e.comparison = comparison.value;
        e.comparison_state = comparison;
      }
    }
    jobs.clear();
    for (const auto& [node, all] : by_node) {
      std::vector<size_t> ids;
      for (auto i : all) if (events[i].active) ids.push_back(i);
      stats["next_steps"] += all.size();
      if (!ids.empty()) stats["full_calls"] += options_.packed ? 1 : ids.size();
      if (options_.packed && replay) stats["semantic_full_replays"] += ids.size();
      if (options_.packed && !model_.nodes[node].full_kernel->joint_batch()) stats["full_scalar_fallback_steps"] += ids.size();
      jobs.push_back([&, node, all, ids] {
        const auto& w = model_.nodes[node];
        for (auto i : all) {
          auto& e = events[i];
          e.next_state = evaluate_next(graph_.nodes[node], w, {e.old, e.comparison_state, e.time, e.local_content(), e.active, e.control});
          e.next = e.next_state.value;
        }
        if (!ids.empty()) evaluate_full(graph_, model_, events, ids, options_, options_.packed);
      });
    }
    pool_.run(std::move(jobs));
    for (auto& event : events) {
      q.states[{event.batch, event.node}] = event.next_state;
      if (event.active) {
        deliver(graph_, model_, event, [&](const Atom& a) {
          queue[a.time].push_back(a);
          if (options_.trace) result.messages.push_back(a);
          ++stats["visited_edges"];
        }, [&](const Output& output) { result.outputs.push_back(output); });
      }
      if (options_.trace) result.trace.push_back(std::move(event));
    }
  }
  q.cut = stop;
  std::sort(result.messages.begin(), result.messages.end(), [&](const auto& a, const auto& b) {
    return std::tie(a.position, a.batch, graph_.edges[a.source].source, a.source)
         < std::tie(b.position, b.batch, graph_.edges[b.source].source, b.source);
  });
  std::sort(result.outputs.begin(), result.outputs.end(), [&](const auto& a, const auto& b) {
    return std::tie(a.time, a.batch, graph_.outputs[a.port], a.port)
         < std::tie(b.time, b.batch, graph_.outputs[b.port], b.port);
  });
  return result;
}
}  // namespace tide
