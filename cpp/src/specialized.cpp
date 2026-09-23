#include "tide/specialized.h"
#include "tide/ops.h"
#include "tide/kernel.h"
#include "tide/delivery.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace tide {
Specialized::Specialized(Graph g, Model m, Options options, std::string topology)
    : graph_(std::move(g)), model_(std::move(m)), options_(options), topology_(std::move(topology)), pool_(options.workers) {
  if (options.parallel_regions || options.compact_events || options.defer_state_release || options.packed_sources || options.batch_next)
    throw std::invalid_argument("streaming optimizations require Streaming");
  graph_.compile(); configure_model(graph_, model_); validate_model(graph_, model_);
  const Index n = graph_.nodes.size();
  auto require = [](bool ok) { if (!ok) throw std::invalid_argument("specialization topology/options mismatch"); };
  require(topology_ == "self_loop" || topology_ == "chain");
  require(graph_.inputs == std::vector<Index>{0} && graph_.outputs == std::vector<Index>{n - 1});
  require(static_cast<Index>(graph_.regions.size()) == n);
  for (Index v = 0; v < n; ++v) require(graph_.nodes[v].region == v);
  require(static_cast<Index>(graph_.edges.size()) == (topology_ == "self_loop" ? 1 : n - 1));
  if (topology_ == "self_loop") require(n == 1 && graph_.edges[0].source == 0 && graph_.edges[0].target == 0);
  else for (Index v = 0; v < n - 1; ++v) require(graph_.edges[v].source == v && graph_.edges[v].target == v + 1);
  require(options.mode == "hard" || options.mode == "hst" || options.mode == "softp");
  require(std::isfinite(options.zeta));
}
Result Specialized::run(const Continuation& initial, const std::vector<External>& xs, Index stop, Index seal) {
  Result result; auto& q = result.continuation; q = initial;
  auto atoms = validate_window(graph_, model_, q, xs, stop, seal);
  atoms.insert(atoms.end(), q.pending.begin(), q.pending.end());
  Fibers inbox;
  for (const auto& a : atoms) inbox[{a.batch, a.node, a.time}].push_back(a);
  const Index n = graph_.nodes.size();
  auto execute = [&](Index node, const std::vector<Frame>& frames) {
    // Local region-block formulas are shared; the fixed topology owns scheduling
    // and propagation and never invokes Streaming, Frontier or their planner.
    auto events = evaluate_block(graph_, model_, q, frames, inbox, options_, pool_, result.stats);
    for (auto& e : events) {
      deliver(graph_, model_, e, [&](const Atom& a) {
        inbox[{a.batch, a.node, a.time}].push_back(a);
        if (options_.trace) result.messages.push_back(a);
      }, [&](const Output& output) { result.outputs.push_back(output); });
      ++result.stats["candidate_events"];
      if (options_.trace) result.trace.push_back(std::move(e));
    }
  };
  if (topology_ == "self_loop") {
    for (Index time = q.cut; time < stop; ++time) {
      std::vector<Frame> frames;
      for (Index b = 0; b < q.batch_size; ++b)
        if (inbox.count({b, 0, time})) frames.push_back({b, 0, time, {0}, {}});
      execute(0, frames);
    }
  } else {
    for (Index node = 0; node < n; ++node) {
      std::vector<Frame> frames;
      for (const auto& [coordinate, bucket] : inbox) {
        auto [b, v, t] = coordinate;
        if (v == node && t >= q.cut && t < stop) frames.push_back({b, node, t, {node}, {}});
      }
      std::sort(frames.begin(), frames.end(), [](const auto& a, const auto& b) { return std::tie(a.time, a.batch) < std::tie(b.time, b.batch); });
      execute(node, frames);
    }
  }
  q.pending.clear(); q.cut = stop;
  for (const auto& [coordinate, bucket] : inbox)
    for (const auto& a : bucket) if (a.kind == 1 && a.time >= stop) q.pending.push_back(a);
  std::sort(q.pending.begin(), q.pending.end(), [](const auto& a, const auto& b) { return a.key() < b.key(); });
  std::sort(result.trace.begin(), result.trace.end(), [](const auto& a, const auto& b) {
    return std::tie(a.time, a.batch, a.node) < std::tie(b.time, b.batch, b.node);
  });
  std::sort(result.messages.begin(), result.messages.end(), [&](const auto& a, const auto& b) {
    return std::tie(a.position, a.batch, graph_.edges[a.source].source, a.source)
         < std::tie(b.position, b.batch, graph_.edges[b.source].source, b.source);
  });
  std::sort(result.outputs.begin(), result.outputs.end(), [](const auto& a, const auto& b) {
    return std::tie(a.time, a.batch) < std::tie(b.time, b.batch);
  });
  return result;
}
}  // namespace tide
