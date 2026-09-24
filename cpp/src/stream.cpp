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
#include "tide/stream_profile.h"
#include "stream_support.h"
#include "tide/operator_work.h"
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
  validate_full_autograd(model_, options_);
  if (options_.mode != "hard" && options_.mode != "softp" && options_.mode != "hst")
    throw std::invalid_argument("invalid emit mode");
  if (!std::isfinite(options_.zeta)) throw std::invalid_argument("nonfinite zeta");
  if (options_.defer_state_release && !options_.compact_events)
    throw std::invalid_argument("deferred state release requires compact events");
  if ((options_.packed_sources || options_.batch_next) && !options_.packed)
    throw std::invalid_argument("packed transport requires packed Streaming");
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
    StreamProfile profile(options_.profile, stats);
    const Index time = queue.begin()->first;
    auto arrived = std::move(queue.begin()->second);
    queue.erase(queue.begin());
    ++stats["logical_times"];
    stats["source_rows"] += arrived.size();
    std::map<Owner, std::vector<Atom>> fibers;
    std::vector<Event> events;
    NodeEvents by_node;
    RegionEvents by_region;
    if (options_.compact_events) events = compact_stream_events(graph_, model_, q, arrived, by_node, by_region);
    else for (const auto& a : arrived) fibers[{a.batch, a.node}].push_back(a);
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
    profile.phase("profile_update_ns");
    std::vector<std::function<void()>> jobs;
    for (const auto& [node, ids] : by_node) {
      if (!graph_.nodes[node].identity) stats["body_candidate_events"] += ids.size();
      stats["max_node_batch"] = std::max<Index>(stats["max_node_batch"], ids.size());
      stats["update_calls"] += options_.packed ? 1 : ids.size();
      stats["read_calls"] += options_.packed ? 1 : ids.size();
      if (options_.packed && replay) stats["semantic_read_replays"] += ids.size();
      if (options_.packed && !model_.nodes[node].read_kernel->joint_batch()) stats["read_scalar_batch_steps"] += ids.size();
      if (options_.packed && replay) stats["semantic_state_replays"] += ids.size();
      if (model_.nodes[node].kernel->scalar_policy_fallback()) {
        if (!options_.packed) stats["fiber_policy_scalar_events"] += ids.size();
        else if (replay) stats["fiber_policy_semantic_replays"] += ids.size();
      }
      if (options_.packed && !model_.nodes[node].kernel->joint_batch()) stats["state_scalar_batch_steps"] += ids.size();
      stats["aggregate_calls"] += options_.packed ? 1 : ids.size();
      if (options_.packed && replay) stats["semantic_aggregate_replays"] += ids.size();
      if (options_.packed && !model_.nodes[node].aggregate_kernel->joint_batch()) stats["aggregate_scalar_fallback_steps"] += ids.size();
      if (options_.packed_sources) {
        if (model_.nodes[node].aggregate_kernel->joint_sources()) stats["packed_source_batches"] += 1;
        else stats["packed_source_fallback_events"] += ids.size();
      }
      jobs.push_back([&, node, ids] {
        const auto& w = model_.nodes[node];
        auto packed_content = evaluate_aggregate(graph_, model_, events, ids, options_.packed, options_.packed_sources);
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
            auto h = packed_content.defined() ? packed_content : at::stack(content);
            states = w.kernel->batch(w, old, h, times, content_views);
            if (states.size() != ids.size()) throw std::invalid_argument("state batch changed event count");
          }
          for (size_t k = 0; k < ids.size(); ++k) {
            auto& e = events[ids[k]];
            e.proposed_state = states[k];
            if (replay) {
              work::StateReplayTimer replay_timer;
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
    profile.phase("profile_select_ns");
    if (options_.parallel_regions) {
      parallel_stream_regions(graph_, model_, q, events, by_region, pool_, options_.workers, options_.trace);
      stats["region_steps"] += by_region.size();
    } else for (const auto& [owner, ids] : by_region) {
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
    profile.phase("profile_full_ns");
    jobs.clear();
    for (const auto& [node, all] : by_node) {
      std::vector<size_t> ids;
      for (auto i : all) if (events[i].active) ids.push_back(i);
      stats["selected_events"] += ids.size();
      if (!graph_.nodes[node].identity) stats["body_selected_events"] += ids.size();
      stats["max_full_batch"] = std::max<Index>(stats["max_full_batch"], ids.size());
      stats["next_steps"] += all.size();
      if (options_.batch_next) {
        const auto& w = model_.nodes[node];
        if (w.next_kernel->joint_batch()) {
          ++stats["next_batches"];
          if (replay) stats["semantic_next_replays"] += all.size();
          if (graph_.nodes[node].clear && !ids.empty()) {
            if (w.kernel->joint_reset_batch()) ++stats["next_reset_batches"];
            else stats["next_reset_scalar_steps"] += ids.size();
          }
        } else stats["next_scalar_fallback_steps"] += all.size();
      }
      if (!ids.empty()) stats["full_calls"] += options_.packed ? 1 : ids.size();
      if (options_.packed && replay) stats[options_.full_autograd == "batched" ? "batched_full_events" : "semantic_full_replays"] += ids.size();
      if (options_.packed && !model_.nodes[node].full_kernel->joint_batch()) stats["full_scalar_fallback_steps"] += ids.size();
      jobs.push_back([&, node, all, ids] {
        const auto& w = model_.nodes[node];
        std::vector<State> next;
        if (options_.batch_next) {
          std::vector<NextInput> requests;
          for (auto i : all) {
            const auto& e = events[i];
            requests.push_back({e.old, e.comparison_state, e.time, e.local_content(), e.active, e.control});
          }
          next = evaluate_next_batch(graph_.nodes[node], w, requests);
        }
        size_t row = 0;
        for (auto i : all) {
          auto& e = events[i];
          e.next_state = options_.batch_next ? std::move(next[row++]) :
            evaluate_next(graph_.nodes[node], w, {e.old, e.comparison_state, e.time, e.local_content(), e.active, e.control});
          e.next = e.next_state.value;
        }
        if (!ids.empty()) evaluate_full(graph_, model_, events, ids, options_, options_.packed);
      });
    }
    pool_.run(std::move(jobs));
    profile.phase("profile_commit_ns");
    for (auto& event : events) {
      if (options_.compact_events && !options_.trace) {
        auto& state = q.states[{event.batch, event.node}];
        if (options_.defer_state_release) std::swap(state, event.next_state);
        else state = std::move(event.next_state);
      }
      else q.states[{event.batch, event.node}] = event.next_state;
      if (event.active) {
        deliver(graph_, model_, event, [&](const Atom& a) {
          queue[a.time].push_back(a);
          if (options_.trace) result.messages.push_back(a);
          ++stats["visited_edges"];
        }, [&](const Output& output) { result.outputs.push_back(output); });
      }
      if (options_.trace) result.trace.push_back(std::move(event));
    }
    profile.phase("profile_cleanup_ns");
    if (options_.compact_events && !options_.trace) release_stream_events(events, pool_, options_.workers);
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
