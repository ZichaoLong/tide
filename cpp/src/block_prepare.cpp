#include "tide/block.h"
#include "tide/autograd.h"
#include "tide/read.h"
#include "tide/next.h"
#include <ATen/core/grad_mode.h>
#include <algorithm>
#include <stdexcept>

namespace tide {
void prefill_states(const Graph& g, const Model& m, const Continuation& q, const Options& options,
                    NodePool& pool, std::vector<Event>& events,
                    const std::map<Owner, std::vector<size_t>>& by_sequence, std::map<std::string, Index>& stats) {
  struct Task { Index node; PackedSequence batch; std::vector<State> old; std::vector<size_t> ids; PackedStates result; };
  std::vector<Task> tasks;
  const bool replay = at::GradMode::is_enabled();
  std::map<Index, size_t> by_node;
  for (const auto& [owner, ids] : by_sequence) {
    const auto node = owner.second;
    const char* fallback = nullptr;
    if (!options.prefill) fallback = "state_prefill_disabled";
    else if (!m.nodes[node].kernel->exact_sequence()) fallback = "state_prefill_no_sequence_contract";
    else if (!g.regions[g.nodes[node].region].observe_all) fallback = "state_prefill_selected_adoption";
    else if (g.nodes[node].clear) fallback = "state_prefill_selected_clear";
    else if (!m.nodes[node].next_kernel->comparison_identity()) fallback = "state_prefill_blocked_next";
    if (fallback) { stats[fallback] += ids.size(); stats["state_prefill_fallback_events"] += ids.size(); continue; }
    size_t index;
    if (options.packed && by_node.count(node)) index = by_node.at(node);
    else { index = tasks.size(); by_node[node] = index; tasks.push_back(Task{node}); }
    auto& task = tasks[index];
    auto it = q.states.find(owner);
    task.old.push_back(it == q.states.end() ? m.nodes[node].kernel->initial(m.nodes[node]) : it->second);
    task.batch.owners.push_back(owner);
    task.ids.insert(task.ids.end(), ids.begin(), ids.end());
    task.batch.offsets.push_back(task.ids.size());
    ++stats["state_blocks"];  // semantic sequences; calls below measures actual grouped execution
  }
  std::vector<std::function<void()>> jobs;
  for (size_t i = 0; i < tasks.size(); ++i) jobs.push_back([&, i] {
    auto& task = tasks[i]; auto& batch = task.batch;
    std::vector<Tensor> values;
    for (auto j : task.ids) {
      values.push_back(events[j].content); batch.times.push_back(events[j].time); batch.views.push_back(events[j].local_content());
    }
    const auto& w = m.nodes[task.node];
    {
      at::NoGradGuard guard;
      batch.contents = at::stack(values);
      task.result = w.kernel->packed_sequence(w, task.old, batch);
      const auto& states = task.result.states;
      if (states.size() != task.ids.size()) throw std::invalid_argument("packed kernel returned incorrect state count");
    }
    auto& states = task.result.states;
    for (size_t s = 0; s < task.old.size(); ++s) {
      auto previous = task.old[s];
      for (Index j = batch.offsets[s]; j < batch.offsets[s + 1]; ++j) {
        auto& e = events[task.ids[j]];
        e.old = previous;
        e.proposed_state = states[j];
        if (replay) {
          auto reference = w.kernel->step(w, previous, e.local_content(), e.time);
          e.proposed_state = semantic_state(e.proposed_state, reference);
        }
        e.proposal = e.proposed_state.value;
        previous = e.proposed_state;
      }
    }
    evaluate_read(g, m, events, task.ids, true);
  });
  pool.run(std::move(jobs));
  for (const auto& task : tasks) {
    if (replay) {
      stats["semantic_state_replays"] += task.ids.size();
      if (m.nodes[task.node].kernel->scalar_policy_fallback()) stats["fiber_policy_semantic_replays"] += task.ids.size();
      stats["semantic_read_replays"] += task.ids.size();
    }
    ++stats["read_calls"];
    if (!m.nodes[task.node].read_kernel->joint_batch()) stats["read_scalar_batch_steps"] += task.ids.size();
    stats["state_sequence_calls"] += task.result.calls;
    stats["state_scalar_sequence_steps"] += task.result.scalar_steps;
    stats["max_state_batch"] = std::max(stats["max_state_batch"], task.result.max_batch);
    stats["max_state_sequence"] = std::max(stats["max_state_sequence"], task.result.max_length);
    stats["attention_score_elements"] += task.result.score_elements;
  }
}
}  // namespace tide
