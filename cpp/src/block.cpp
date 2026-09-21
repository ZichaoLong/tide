#include "tide/frontier.h"
#include "tide/ops.h"
#include "tide/block.h"
#include "tide/autograd.h"
#include "tide/full.h"
#include "tide/aggregate.h"
#include <ATen/core/grad_mode.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace tide {
namespace {
State old_state(const Continuation& q, const Model& m, Index batch, Index node) {
  auto it = q.states.find({batch, node});
  return it == q.states.end() ? m.nodes[node].kernel->initial(m.nodes[node]) : it->second;
}
}  // namespace
std::vector<Event> evaluate_block(const Graph& g, const Model& m, Continuation& q, const std::vector<Frame>& frames,
                                 const Fibers& fibers, const Options& options, NodePool& pool,
                                 std::map<std::string, Index>& stats) {
  std::vector<Event> events;
  std::map<Index, std::vector<size_t>> by_node;
  std::map<Owner, std::vector<size_t>> by_sequence;
  std::map<Owner, std::vector<size_t>> by_frame;  // batch,time (one region per block)
  for (const auto& frame : frames) {
    for (auto node : frame.nodes) {
      auto it = fibers.find({frame.batch, node, frame.time});
      if (it == fibers.end() || it->second.empty()) continue;
      Event e;
      e.batch = frame.batch; e.node = node; e.time = frame.time; e.fiber = it->second;
      std::sort(e.fiber.begin(), e.fiber.end(), [](const auto& a, const auto& b) { return a.key() < b.key(); });
      by_node[node].push_back(events.size());
      by_sequence[{frame.batch, node}].push_back(events.size());
      by_frame[{frame.batch, frame.time}].push_back(events.size());
      events.push_back(std::move(e));
    }
  }
  std::vector<std::function<void()>> jobs;
  for (const auto& [node, ids] : by_node) {
    ++stats["aggregate_calls"];
    if (at::GradMode::is_enabled()) stats["semantic_aggregate_replays"] += ids.size();
    if (!m.nodes[node].aggregate_kernel->joint_batch()) stats["aggregate_scalar_fallback_steps"] += ids.size();
    jobs.push_back([&, ids] { evaluate_aggregate(g, m, events, ids, true); });
  }
  pool.run(std::move(jobs));
  prefill_states(g, m, q, options, pool, events, by_sequence, stats);
  for (const auto& frame : frames) {
    const auto& ids = by_frame[{frame.batch, frame.time}];
    if (ids.empty()) continue;
    const auto& region = g.regions[frame.region];
    jobs.clear();
    for (auto i : ids) {
      if (events[i].proposal.defined()) continue;
      ++stats["state_steps"];
      events[i].old = old_state(q, m, events[i].batch, events[i].node);
      jobs.push_back([&, i] {
        auto& e = events[i]; const auto& w = m.nodes[e.node];
        e.proposed_state = w.kernel->step(w, e.old, e.local_content(), e.time);
        e.proposal = e.proposed_state.value;
        e.descriptor = w.kernel->read(w, e.old, e.proposed_state, e.local_content(), e.time);
      });
    }
    pool.run(std::move(jobs));
    auto& history = q.history[{frame.batch, frame.region}];
    std::vector<std::tuple<Index, double, Index, size_t>> ranking;
    std::vector<Tensor> desc;
    for (auto i : ids) {
      const auto& e = events[i];
      auto count = history.find(e.node); auto score = e.descriptor.item<double>();
      if (!std::isfinite(score)) throw std::invalid_argument("nonfinite selector score");
      ranking.emplace_back(region.count_priority && count != history.end() ? count->second : 0, -score, e.node, i);
      desc.push_back(e.descriptor);
    }
    std::sort(ranking.begin(), ranking.end());
    for (Index i = 0; i < std::min<Index>(region.budget, ranking.size()); ++i) {
      auto& e = events[std::get<3>(ranking[i])]; e.active = true; ++history[e.node];
    }
    auto controls = at::softmax(at::stack(desc), 0);
    for (size_t j = 0; j < ids.size(); ++j) {
      auto& e = events[ids[j]]; e.control = controls[j];
      auto cmp = region.observe_all || e.active ? e.proposed_state : old_state(q, m, e.batch, e.node);
      e.comparison = cmp.value;
      e.comparison_state = cmp;
      e.next_state = g.nodes[e.node].clear && e.active ? m.nodes[e.node].kernel->reset(cmp) : cmp;
      e.next = e.next_state.value;
      q.states[{e.batch, e.node}] = e.next_state;
      if (options.trace) e.history = history;
    }
  }
  jobs.clear();
  const bool replay = at::GradMode::is_enabled();
  for (const auto& [node, all] : by_node) {
    std::vector<size_t> ids;
    for (auto i : all) if (events[i].active) ids.push_back(i);
    if (ids.empty()) continue;
    ++stats["full_blocks"];
    if (replay) stats["semantic_full_replays"] += ids.size();
    if (!m.nodes[node].full_kernel->joint_batch()) stats["full_scalar_fallback_steps"] += ids.size();
    jobs.push_back([&, ids] { evaluate_full(g, m, events, ids, options, true); });
  }
  pool.run(std::move(jobs));
  return events;
}
}  // namespace tide
