// Causal frame waves inside a certified frontier block. Independent sample/
// region owners may share a wave; each owner's logical times remain ordered.
#include "tide/block.h"
#include "tide/next.h"
#include "tide/read.h"
#include "tide/region.h"
#include "stream_support.h"
#include <ATen/core/grad_mode.h>
#include <algorithm>

namespace tide {
void advance_block_events(const Graph& g, const Model& m, Continuation& q,
                          std::vector<Event>& events, const std::vector<std::vector<size_t>>& waves,
                          const Options& options, NodePool& pool, std::map<std::string, Index>& stats) {
  for (const auto& ids : waves) {
    if (ids.empty()) continue;
    NodeEvents nodes;
    RegionEvents regions;
    std::vector<std::function<void()>> jobs;
    for (auto i : ids) {
      auto& e = events[i];
      nodes[e.node].push_back(i); regions[{e.batch, g.nodes[e.node].region}].push_back(i);
      if (e.proposal.defined()) continue;
      ++stats["state_steps"]; ++stats["read_calls"];
      const auto old = q.states.find({e.batch, e.node});
      e.old = old == q.states.end() ? m.nodes[e.node].kernel->initial(m.nodes[e.node]) : old->second;
      jobs.push_back([&, i] {
        auto& e = events[i]; const auto& w = m.nodes[e.node];
        e.proposed_state = w.kernel->step(w, e.old, e.local_content(), e.time);
        e.proposal = e.proposed_state.value;
        evaluate_read(g, m, events, {i}, false);
      });
    }
    pool.run(std::move(jobs));
    if (options.parallel_regions) {
      parallel_stream_regions(g, m, q, events, regions, pool, options.workers, options.trace);
      ++stats["region_parallel_waves"];
      stats["max_region_wave"] = std::max<Index>(stats["max_region_wave"], regions.size());
    } else for (const auto& [owner, candidates] : regions) {
      select_events(g, m, q, events, candidates, options.trace);
      for (auto i : candidates) {
        auto& e = events[i];
        e.comparison_state = g.regions[owner.second].observe_all || e.active ? e.proposed_state : e.old;
        e.comparison = e.comparison_state.value;
      }
    }
    stats["region_steps"] += regions.size();
    stats["next_steps"] += ids.size();
    jobs.clear();
    if (options.batch_next) {
      for (const auto& [node, all] : nodes) {
        const auto& w = m.nodes[node];
        if (w.next_kernel->joint_batch()) {
          ++stats["next_batches"];
          stats["max_next_batch"] = std::max<Index>(stats["max_next_batch"], all.size());
          if (at::GradMode::is_enabled()) stats["semantic_next_replays"] += all.size();
          if (g.nodes[node].clear) {
            const auto selected = std::count_if(all.begin(), all.end(), [&](auto i) { return events[i].active; });
            if (selected && w.kernel->joint_reset_batch()) ++stats["next_reset_batches"];
            else stats["next_reset_scalar_steps"] += selected;
          }
        } else stats["next_scalar_fallback_steps"] += all.size();
        jobs.push_back([&, node, all] {
          std::vector<NextInput> requests;
          for (auto i : all) {
            const auto& e = events[i];
            requests.push_back({e.old, e.comparison_state, e.time, e.local_content(), e.active, e.control});
          }
          auto next = evaluate_next_batch(g.nodes[node], m.nodes[node], requests);
          for (size_t j = 0; j < all.size(); ++j) {
            auto& e = events[all[j]]; e.next_state = std::move(next[j]); e.next = e.next_state.value;
          }
        });
      }
    } else for (auto i : ids) jobs.push_back([&, i] {
      auto& e = events[i];
      e.next_state = evaluate_next(g.nodes[e.node], m.nodes[e.node],
                                  {e.old, e.comparison_state, e.time, e.local_content(), e.active, e.control});
      e.next = e.next_state.value;
    });
    pool.run(std::move(jobs));
    for (auto i : ids) {
      auto& e = events[i];
      auto& state = q.states[{e.batch, e.node}];
      if (options.compact_events && !options.trace) {
        if (options.defer_state_release) {
          stats["deferred_state_releases"] += state.value.defined();
          std::swap(state, e.next_state);  // Event cleanup retires the old owner.
        } else state = std::move(e.next_state);
      } else state = e.next_state;
    }
  }
}
}  // namespace tide
