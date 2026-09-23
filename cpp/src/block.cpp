#include "tide/frontier.h"
#include "tide/ops.h"
#include "tide/block.h"
#include "tide/autograd.h"
#include "tide/full.h"
#include "tide/aggregate.h"
#include "tide/read.h"
#include "tide/next.h"
#include "tide/region.h"
#include <ATen/core/grad_mode.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace tide {
std::vector<Event> evaluate_block(const Graph& g, const Model& m, Continuation& q, const std::vector<Frame>& frames,
                                 Fibers& fibers, const Options& options, NodePool& pool,
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
      e.batch = frame.batch; e.node = node; e.time = frame.time;
      stats["source_rows"] += it->second.size();
      if (options.compact_events) {
        e.fiber = std::move(it->second); fibers.erase(it); ++stats["consumed_fibers"];
      } else e.fiber = it->second;
      std::sort(e.fiber.begin(), e.fiber.end(), [](const auto& a, const auto& b) { return a.key() < b.key(); });
      by_node[node].push_back(events.size());
      by_sequence[{frame.batch, node}].push_back(events.size());
      by_frame[{frame.batch, frame.time}].push_back(events.size());
      events.push_back(std::move(e));
    }
  }
  std::vector<std::function<void()>> jobs;
  for (const auto& [node, ids] : by_node) {
    if (!g.nodes[node].identity) stats["body_candidate_events"] += ids.size();
    ++stats["aggregate_calls"];
    if (at::GradMode::is_enabled()) stats["semantic_aggregate_replays"] += ids.size();
    if (!m.nodes[node].aggregate_kernel->joint_batch()) stats["aggregate_scalar_fallback_steps"] += ids.size();
    if (options.packed_sources) {
      if (m.nodes[node].aggregate_kernel->joint_sources()) ++stats["packed_source_batches"];
      else stats["packed_source_fallback_events"] += ids.size();
    }
    jobs.push_back([&, ids] { evaluate_aggregate(g, m, events, ids, true, options.packed_sources); });
  }
  pool.run(std::move(jobs));
  prefill_states(g, m, q, options, pool, events, by_sequence, stats);
  std::vector<std::vector<size_t>> waves;
  Index previous_time = -1;
  for (const auto& frame : frames) {
    const auto& ids = by_frame[{frame.batch, frame.time}];
    if (ids.empty()) continue;
    if (waves.empty() || previous_time != frame.time || (!options.batch_next && !options.parallel_regions))
      waves.emplace_back();
    waves.back().insert(waves.back().end(), ids.begin(), ids.end());
    previous_time = frame.time;
  }
  advance_block_events(g, m, q, events, waves, options, pool, stats);
  jobs.clear();
  const bool replay = at::GradMode::is_enabled();
  for (const auto& [node, all] : by_node) {
    std::vector<size_t> ids;
    for (auto i : all) if (events[i].active) ids.push_back(i);
    if (ids.empty()) continue;
    stats["selected_events"] += ids.size();
    if (!g.nodes[node].identity) stats["body_selected_events"] += ids.size();
    stats["max_full_batch"] = std::max<Index>(stats["max_full_batch"], ids.size());
    ++stats["full_blocks"];
    if (replay) stats["semantic_full_replays"] += ids.size();
    if (!m.nodes[node].full_kernel->joint_batch()) stats["full_scalar_fallback_steps"] += ids.size();
    jobs.push_back([&, ids] { evaluate_full(g, m, events, ids, options, true); });
  }
  pool.run(std::move(jobs));
  return events;
}
}  // namespace tide
