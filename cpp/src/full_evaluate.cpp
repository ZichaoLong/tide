#include "tide/operator_work.h"
#include "tide/operator_profile.h"
#include "tide/full.h"
#include "tide/autograd.h"
#include <ATen/core/grad_mode.h>
#include <stdexcept>

namespace tide {
namespace {
void validate(const FullResult& result, const FullInput& input, Index slots) {
  auto tensor = [&](const Tensor& t) {
    if (!t.defined() || t.sizes() != input.content.value.sizes() || t.scalar_type() != input.content.value.scalar_type()
        || t.device() != input.content.value.device()) throw std::invalid_argument("Full returned incompatible tensor metadata");
  };
  if (result.value.defined()) tensor(result.value);
  Index previous = -1;
  for (const auto& emission : result.emitted) {
    if (emission.slot <= previous || emission.slot >= slots) throw std::invalid_argument("Full returned invalid output slots");
    previous = emission.slot; tensor(emission.value);
  }
}
FullResult bind(const FullResult& packed, const FullResult& reference) {
  if (packed.value.defined() != reference.value.defined() || packed.emitted.size() != reference.emitted.size())
    throw std::invalid_argument("Full batch changed output presence");
  FullResult result;
  if (packed.value.defined()) result.value = semantic_value(packed.value, reference.value);
  for (size_t j = 0; j < packed.emitted.size(); ++j) {
    const auto& p = packed.emitted[j]; const auto& r = reference.emitted[j];
    if (p.slot != r.slot) throw std::invalid_argument("Full batch changed output presence");
    const auto same = reference.value.defined() && r.value.unsafeGetTensorImpl() == reference.value.unsafeGetTensorImpl();
    result.emitted.push_back({p.slot, semantic_value(p.value, same ? result.value : r.value)});
  }
  return result;
}
}  // namespace
void validate_full_autograd(const Model& model, const Options& options) {
  if (options.full_autograd != "replay" && options.full_autograd != "batched")
    throw std::invalid_argument("unknown Full autograd policy");
  if (options.full_autograd == "batched") {
    if (!options.packed) throw std::invalid_argument("batched Full autograd requires packed execution");
    for (const auto& w : model.nodes) if (!w.full_kernel->batched_autograd())
      throw std::invalid_argument("Full program has no batched autograd implementation");
  }
}
std::vector<FullResult> FullKernel::batch_grad(const NodeWeights&, const std::vector<FullInput>&,
                                             Index, const Options&) const {
  throw std::invalid_argument("Full program has no batched autograd implementation");
}
void evaluate_full(const Graph& g, const Model& m, std::vector<Event>& events, const std::vector<size_t>& ids,
                   const Options& options, bool packed) {
  if (ids.empty()) return;
  op_profile::Scope profile(op_profile::FullOther);
  const auto node = events[ids.front()].node;
  const auto slots = g.outgoing_ports.offsets[node + 1] - g.outgoing_ports.offsets[node];
  const auto& w = m.nodes[node];
  std::vector<FullInput> inputs;
  for (auto i : ids) {
    const auto& e = events[i]; inputs.push_back({&e.comparison_state, e.time, e.local_content(), e.control});
  }
  std::vector<FullResult> results;
  if (packed && at::GradMode::is_enabled() && options.full_autograd == "batched") {
    results = w.full_kernel->batch_grad(w, inputs, slots, options);
    if (results.size() != ids.size()) throw std::invalid_argument("Full batch changed event count");
    for (size_t j = 0; j < ids.size(); ++j) validate(results[j], inputs[j], slots);
  } else if (packed) {
    {
      at::NoGradGuard guard;
      results = w.full_kernel->batch(w, inputs, slots, options);
    }
    if (results.size() != ids.size()) throw std::invalid_argument("Full batch changed event count");
    for (size_t j = 0; j < ids.size(); ++j) {
      validate(results[j], inputs[j], slots);
      if (at::GradMode::is_enabled()) {
        work::StateReplayTimer replay_timer(work::FullReplayNs);
        auto reference = w.full_kernel->step(w, inputs[j], slots, options);
        validate(reference, inputs[j], slots);
        results[j] = bind(results[j], reference);
      }
    }
  } else {
    for (const auto& input : inputs) {
      auto result = w.full_kernel->step(w, input, slots, options); validate(result, input, slots);
      results.push_back(std::move(result));
    }
  }
  for (size_t j = 0; j < ids.size(); ++j) {
    auto& e = events[ids[j]]; e.full = results[j].value; e.emitted = std::move(results[j].emitted);
  }
}
}  // namespace tide
