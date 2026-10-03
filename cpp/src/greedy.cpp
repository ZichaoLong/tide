#include "tide/greedy.h"
#include "tide/aggregate.h"
#include "tide/full.h"
#include "tide/kernel.h"
#include "tide/ops.h"
#include "tide/ownership.h"
#include "tide/delivery.h"
#include "greedy_plan.h"
#include "stream_support.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace tide {
Greedy::Greedy(Graph g, Model m, Options options)
    : graph_(std::move(g)), model_(std::move(m)), options_(options), pool_(options.workers) {
  if (options.profile) throw std::invalid_argument("phase profiling requires Streaming");
  if (options.defer_state_release && !options.compact_events)
    throw std::invalid_argument("deferred state release requires compact events");
  if ((options.packed_sources || options.batch_next) && !options.packed)
    throw std::invalid_argument("packed transport requires packed execution");
  graph_.compile(); configure_model(graph_, model_); validate_model(graph_, model_);
  pool_.set_devices(model_devices(model_));
  validate_full_autograd(model_, options); validate_aggregate_autograd(model_, options);
  if (options.mode != "hard" && options.mode != "hst" && options.mode != "softp")
    throw std::invalid_argument("invalid emit mode");
  if (options.max_events < 1 || !std::isfinite(options.zeta))
    throw std::invalid_argument("invalid greedy options");
}
Result Greedy::run(const Continuation& initial, const std::vector<External>& external, Index stop, Index seal) {
  Result result; auto& q = result.continuation; q = initial;
  auto atoms = validate_window(graph_, model_, q, external, stop, seal);
  atoms.insert(atoms.end(), q.pending.begin(), q.pending.end());
  Fibers fibers;
  for (const auto& a : atoms) fibers[{a.batch, a.node, a.time}].push_back(a);
  ClosurePlan plan(graph_);
  auto& stats = result.stats;
  stats = {{"greedy_stages",0}, {"greedy_relaxed_edges",0}, {"region_blocks",0},
           {"candidate_events",0}, {"visited_edges",0}, {"max_live_fibers",0}};
  while (true) {
    stats["max_live_fibers"] = std::max<Index>(stats["max_live_fibers"], fibers.size());
    const auto blocks = plan.ready(graph_, fibers, stop, options_.max_events, stats["greedy_relaxed_edges"]);
    if (blocks.empty()) break;
    ++stats["greedy_stages"];
    for (const auto& [region, frames] : blocks) {
      auto events = evaluate_block(graph_, model_, q, frames, fibers, options_, pool_, stats);
      for (const auto& frame : frames)
        for (auto node : frame.nodes) fibers.erase({frame.batch, node, frame.time});
      ++stats["region_blocks"]; stats["candidate_events"] += events.size();
      stats["max_greedy_frames"] = std::max<Index>(stats["max_greedy_frames"], frames.size());
      for (auto& event : events) {
        if (event.active) deliver(graph_, model_, event, [&](const Atom& atom) {
          ++stats["visited_edges"]; fibers[{atom.batch, atom.node, atom.time}].push_back(atom);
          if (options_.trace) result.messages.push_back(atom);
        }, [&](const Output& output) { result.outputs.push_back(output); });
        if (options_.trace) result.trace.push_back(std::move(event));
      }
      if (options_.compact_events && !options_.trace) release_stream_events(events, pool_, options_.workers);
      if (fibers.size() > size_t(options_.max_events)) throw std::invalid_argument("greedy live-fiber capacity exceeded");
    }
  }
  q.pending.clear();
  for (const auto& [coordinate, bucket] : fibers)
    for (const auto& atom : bucket) if (atom.kind == 1) q.pending.push_back(atom);
  q.cut = stop;
  std::sort(q.pending.begin(), q.pending.end(), [](const auto& a, const auto& b) { return a.key() < b.key(); });
  std::sort(result.trace.begin(), result.trace.end(), [](const auto& a, const auto& b) {
    return std::tie(a.time,a.batch,a.node) < std::tie(b.time,b.batch,b.node);
  });
  std::sort(result.messages.begin(), result.messages.end(), [&](const auto& a, const auto& b) {
    return std::tie(a.position,a.batch,graph_.edges[a.source].source,a.source)
         < std::tie(b.position,b.batch,graph_.edges[b.source].source,b.source);
  });
  std::sort(result.outputs.begin(), result.outputs.end(), [&](const auto& a, const auto& b) {
    return std::tie(a.time,a.batch,graph_.outputs[a.port],a.port) < std::tie(b.time,b.batch,graph_.outputs[b.port],b.port);
  });
  return result;
}
}  // namespace tide
