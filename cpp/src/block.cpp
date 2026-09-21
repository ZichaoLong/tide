#include "tide/frontier.h"
#include "tide/ops.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace tide {
namespace {
Tensor scan(Tensor a, Tensor b, const Tensor& initial) {
  const auto n = b.size(0);
  for (Index stride = 1; stride < n; stride *= 2) {
    b = at::cat({b.slice(0, 0, stride), b.slice(0, stride) + a.slice(0, stride) * b.slice(0, 0, n - stride)});
    a = at::cat({a.slice(0, 0, stride), a.slice(0, stride) * a.slice(0, 0, n - stride)});
  }
  return b + a * initial;
}
State old_state(const Continuation& q, const Model& m, Index batch, Index node) {
  auto it = q.states.find({batch, node});
  return it == q.states.end() ? State{at::zeros_like(m.nodes[node].bias)} : it->second;
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
      e.content = aggregate(m, e.fiber);
      by_node[node].push_back(events.size());
      by_sequence[{frame.batch, node}].push_back(events.size());
      by_frame[{frame.batch, frame.time}].push_back(events.size());
      events.push_back(std::move(e));
    }
  }
  std::vector<std::function<void()>> jobs;
  for (const auto& [owner, ids] : by_sequence) {
    const auto node = owner.second;
    if (!options.prefill || !g.regions[g.nodes[node].region].observe_all || g.nodes[node].clear) continue;
    ++stats["state_blocks"];
    auto old = old_state(q, m, owner.first, node);
    jobs.push_back([&, node, ids, old] {
      std::vector<Tensor> values;
      for (auto i : ids) values.push_back(events[i].content);
      auto h = at::stack(values);
      auto proposals = scan(at::sigmoid(m.nodes[node].decay).expand_as(h), h, old.value);
      auto desc = (proposals * m.nodes[node].read).sum(-1);
      for (size_t j = 0; j < ids.size(); ++j) {
        auto& e = events[ids[j]]; e.proposal = proposals[j]; e.descriptor = desc[j];
        e.proposed_state = {e.proposal, e.time, old.observations + static_cast<Index>(j) + 1};
      }
    });
  }
  pool.run(std::move(jobs));
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
        e.proposal = at::sigmoid(w.decay) * e.old.value + e.content;
        e.descriptor = (e.proposal * w.read).sum(-1);
        e.proposed_state = {e.proposal, e.time, e.old.observations + 1};
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
      e.next = g.nodes[e.node].clear && e.active ? cmp.value * 0 : cmp.value;
      q.states[{e.batch, e.node}] = {e.next, cmp.last_time, cmp.observations};
      if (options.trace) e.history = history;
    }
  }
  jobs.clear();
  for (const auto& [node, all] : by_node) {
    std::vector<size_t> ids;
    for (auto i : all) if (events[i].active) ids.push_back(i);
    if (ids.empty()) continue;
    ++stats["full_blocks"];
    jobs.push_back([&, node, ids] {
      std::vector<Tensor> cmp, h, p;
      for (auto i : ids) { cmp.push_back(events[i].comparison); h.push_back(events[i].content); p.push_back(events[i].control); }
      auto values = full(m.nodes[node], at::stack(cmp), at::stack(h), at::stack(p), options);
      for (size_t i = 0; i < ids.size(); ++i) events[ids[i]].full = values[i];
    });
  }
  pool.run(std::move(jobs));
  return events;
}
}  // namespace tide
