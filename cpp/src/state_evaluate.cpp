#include "tide/state_evaluate.h"
#include "tide/kernel.h"
#include "tide/state_vjp.h"
#include "tide/autograd.h"
#include "tide/operator_work.h"
#include <ATen/core/grad_mode.h>
#include <stdexcept>

namespace tide {
void evaluate_state(const Model& model, std::vector<Event>& events, const std::vector<size_t>& ids,
                    bool packed, const Tensor& contents) {
  if (ids.empty()) return;
  const auto& w = model.nodes[events[ids.front()].node];
  if (packed) {
    std::vector<State> old, states;
    std::vector<Tensor> values;
    std::vector<Index> times;
    ContentViews views;
    std::vector<std::map<std::string,Tensor>> prepared;
    bool has_prepared=false;
    for (auto i : ids) {
      const auto& e = events[i];
      old.push_back(e.old); values.push_back(e.content); times.push_back(e.time); views.push_back(e.local_content());
      prepared.push_back(e.state_prepared);has_prepared=has_prepared||!e.state_prepared.empty();
    }
    {
      at::NoGradGuard guard;
      auto h=contents.defined()?contents:at::stack(values);
      states = has_prepared?w.kernel->prepared_batch(w,old,h,times,views,prepared):w.kernel->batch(w,old,h,times,views);
      if (states.size() != ids.size()) throw std::invalid_argument("state batch changed event count");
    }
    const bool batched=at::GradMode::is_enabled() && w.kernel->batched_autograd();
    if (batched) states=state_batch_vjp(w,old,views,times,states,prepared);
    for (size_t j = 0; j < ids.size(); ++j) {
      auto& e = events[ids[j]]; e.proposed_state = std::move(states[j]);
      if (at::GradMode::is_enabled() && !batched) {
        work::StateReplayTimer replay_timer;
        auto reference = w.kernel->step(w, e.old, e.local_content(), e.time);
        e.proposed_state = semantic_state(e.proposed_state, reference);
      }
    }
  } else for (auto i : ids) {
    auto& e = events[i]; e.proposed_state = w.kernel->step(w, e.old, e.local_content(), e.time);
  }
  for (auto i : ids) events[i].proposal = events[i].proposed_state.value;
}
}
