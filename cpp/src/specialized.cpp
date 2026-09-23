#include "tide/specialized.h"
#include "tide/ops.h"
#include "tide/kernel.h"
#include "tide/delivery.h"
#include "stream_support.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace tide {
Specialized::Specialized(Graph g, Model m, Options options, std::string topology)
    : graph_(std::move(g)), model_(std::move(m)), options_(options), topology_(std::move(topology)), pool_(options.workers) {
  if (options.profile) throw std::invalid_argument("phase profiling requires Streaming");
  if (options.defer_state_release && !options.compact_events)
    throw std::invalid_argument("deferred state release requires compact events");
  if ((options.packed_sources || options.batch_next) && !options.packed)
    throw std::invalid_argument("packed transport requires packed execution");
  graph_.compile(); configure_model(graph_, model_); validate_model(graph_, model_);
  const Index n = graph_.nodes.size();
  auto require = [](bool ok) { if (!ok) throw std::invalid_argument("specialization topology/options mismatch"); };
  require(topology_ == "self_loop" || topology_ == "ring" || topology_ == "chain" || topology_ == "diamond");
  require(graph_.inputs == std::vector<Index>{0} && graph_.outputs == std::vector<Index>{n - 1});
  std::vector<std::pair<Index, Index>> expected;
  if (topology_ == "diamond") {
    require(n == 4);
    const auto& nodes = graph_.nodes;
    const bool paired = nodes[1].region == nodes[2].region;
    require(nodes[0].region == 0 && nodes[1].region == 1 && nodes[2].region == (paired ? 1 : 2) &&
            nodes[3].region == (paired ? 2 : 3) && graph_.regions.size() == (paired ? 3 : 4));
    expected = {{0, 1}, {0, 2}, {1, 3}, {2, 3}};
    layers_ = paired ? std::vector<std::vector<Index>>{{0}, {1, 2}, {3}}
                     : std::vector<std::vector<Index>>{{0}, {1}, {2}, {3}};
  } else {
    require(static_cast<Index>(graph_.regions.size()) == n);
    for (Index v = 0; v < n; ++v) {
      require(graph_.nodes[v].region == v); layers_.push_back({v});
    }
    const bool cyclic = topology_ == "self_loop" || topology_ == "ring";
    if (topology_ == "self_loop") require(n == 1);
    if (topology_ == "ring") require(n >= 2);
    for (Index v = 0; v < (cyclic ? n : n - 1); ++v) expected.push_back({v, (v + 1) % n});
  }
  require(graph_.edges.size() == expected.size());
  for (size_t i = 0; i < expected.size(); ++i)
    require(std::make_pair(graph_.edges[i].source, graph_.edges[i].target) == expected[i]);
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
  auto execute = [&](const std::vector<Frame>& frames) {
    // Local region-block formulas are shared; the fixed topology owns scheduling
    // and propagation and never invokes Streaming, Frontier or their planner.
    auto events = evaluate_block(graph_, model_, q, frames, inbox, options_, pool_, result.stats);
    for (auto& e : events) {
      deliver(graph_, model_, e, [&](const Atom& a) {
        ++result.stats["visited_edges"];
        inbox[{a.batch, a.node, a.time}].push_back(a);
        if (options_.trace) result.messages.push_back(a);
      }, [&](const Output& output) { result.outputs.push_back(output); });
      ++result.stats["candidate_events"];
      if (options_.trace) result.trace.push_back(std::move(e));
    }
    if (options_.compact_events && !options_.trace) release_stream_events(events, pool_, options_.workers);
  };
  if (topology_ == "self_loop" || topology_ == "ring") {
    for (Index time = q.cut; time < stop; ++time) {
      for (Index node = 0; node < n; ++node) {
        std::vector<Frame> frames;
        for (Index b = 0; b < q.batch_size; ++b)
          if (inbox.count({b, node, time})) frames.push_back({b, node, time, {node}, {}});
        execute(frames);
      }
    }
  } else {
    for (const auto& layer : layers_) {
      // All predecessors of this fixed layer have published their messages.
      // Group actual fibers by region/time, including paired diamond candidates.
      std::map<std::pair<Index, Index>, std::set<Index>> coordinates;
      std::vector<Frame> frames;
      for (const auto& [coordinate, bucket] : inbox) {
        auto [b, v, t] = coordinate;
        if (std::find(layer.begin(), layer.end(), v) != layer.end() && t >= q.cut && t < stop)
          coordinates[{t, b}].insert(v);
      }
      for (const auto& [coordinate, nodes] : coordinates)
        frames.push_back({coordinate.second, graph_.nodes[layer[0]].region, coordinate.first, nodes, {}});
      execute(frames);
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
