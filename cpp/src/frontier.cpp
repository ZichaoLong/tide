#include "tide/frontier.h"
#include "tide/ops.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace tide {
Frontier::Frontier(Graph g, Model m, Options options)
    : graph_(std::move(g)), model_(std::move(m)), options_(options), pool_(options.workers) {
  graph_.compile(); graph_.topological_order(); validate_model(graph_, model_);
  if (options.mode != "hard" && options.mode != "hst" && options.mode != "softp") throw std::invalid_argument("invalid emit mode");
  if (options.max_events < 1 || !std::isfinite(options.zeta)) throw std::invalid_argument("invalid frontier options");
}
Result Frontier::run(const Continuation& initial, const std::vector<External>& external, Index stop, Index seal) {
  Result result;
  auto& q = result.continuation; q = initial;
  auto atoms = validate_window(graph_, model_, q, external, stop, seal);
  atoms.insert(atoms.end(), q.pending.begin(), q.pending.end());
  const auto frames = plan_frontier(graph_, atoms, q.cut, stop, options_.max_events);
  std::map<Owner, std::vector<Index>> owners;
  std::map<Owner, size_t> cursor;
  for (Index i = 0; i < static_cast<Index>(frames.size()); ++i) owners[{frames[i].batch, frames[i].region}].push_back(i);
  Fibers fibers;
  for (const auto& a : atoms) fibers[{a.batch, a.node, a.time}].push_back(a);
  std::set<Index> done;
  auto& stats = result.stats;
  stats = {{"frontier_stages", 0}, {"region_blocks", 0}, {"state_blocks", 0}, {"state_steps", 0}, {"full_blocks", 0}, {"candidate_events", 0}};
  while (done.size() < frames.size()) {
    std::map<Index, std::vector<Index>> blocks;
    for (const auto& [owner, ids] : owners) {
      auto index = cursor[owner];
      while (index < ids.size()) {
        const auto& deps = frames[ids[index]].dependencies;
        if (!std::includes(done.begin(), done.end(), deps.begin(), deps.end())) break;
        blocks[owner.second].push_back(ids[index]); ++index;
      }
    }
    if (blocks.empty()) throw std::runtime_error("frontier failed to make progress");
    ++stats["frontier_stages"];
    for (auto& [region, ids] : blocks) {
      std::sort(ids.begin(), ids.end(), [&](auto a, auto b) {
        return std::tie(frames[a].time, frames[a].batch) < std::tie(frames[b].time, frames[b].batch);
      });
      std::vector<Frame> ready;
      for (auto i : ids) ready.push_back(frames[i]);
      auto events = evaluate_block(graph_, model_, q, ready, fibers, options_, pool_, stats);
      ++stats["region_blocks"]; stats["candidate_events"] += events.size();
      for (auto& e : events) {
        if (e.active) {
          for (auto j = graph_.csr.offsets[e.node]; j < graph_.csr.offsets[e.node + 1]; ++j) {
            const auto id = graph_.csr.edges[j]; const auto& edge = graph_.edges[id];
            if (e.time > std::numeric_limits<Index>::max() - edge.delay) throw std::overflow_error("logical time overflow");
            Atom a{e.batch, edge.target, e.time + edge.delay, 1, id, e.time, e.full * model_.edge_scale[id]};
            fibers[{a.batch, a.node, a.time}].push_back(a);
            if (options_.trace) result.messages.push_back(a);
          }
          for (auto j = graph_.output_index.offsets[e.node]; j < graph_.output_index.offsets[e.node + 1]; ++j) {
            const auto port = graph_.output_index.edges[j];
            result.outputs.push_back({e.batch, e.time, port, e.full * model_.output_scale[port]});
          }
        }
        if (options_.trace) result.trace.push_back(std::move(e));
      }
      for (auto i : ids) { done.insert(i); ++cursor[{frames[i].batch, frames[i].region}]; }
    }
  }
  q.pending.clear();
  for (const auto& [coordinate, bucket] : fibers)
    for (const auto& a : bucket) if (a.kind == 1 && a.time >= stop) q.pending.push_back(a);
  q.cut = stop;
  std::sort(q.pending.begin(), q.pending.end(), [](const auto& a, const auto& b) { return a.key() < b.key(); });
  std::sort(result.trace.begin(), result.trace.end(), [](const auto& a, const auto& b) {
    return std::tie(a.time, a.batch, a.node) < std::tie(b.time, b.batch, b.node);
  });
  std::sort(result.messages.begin(), result.messages.end(), [&](const auto& a, const auto& b) {
    return std::tie(a.position, a.batch, graph_.edges[a.source].source, a.source)
         < std::tie(b.position, b.batch, graph_.edges[b.source].source, b.source);
  });
  std::sort(result.outputs.begin(), result.outputs.end(), [&](const auto& a, const auto& b) {
    return std::tie(a.time, a.batch, graph_.outputs[a.port], a.port) < std::tie(b.time, b.batch, graph_.outputs[b.port], b.port);
  });
  return result;
}
}  // namespace tide
