#include "tide/aggregate.h"
#include "tide/autograd.h"
#include <ATen/core/grad_mode.h>
#include <set>
#include <stdexcept>

namespace tide {
namespace {
AggregateInput request(const Graph& g, const Model& m, const Event& event) {
  AggregateInput result{event.time, g.incoming_ports.offsets[event.node+1] - g.incoming_ports.offsets[event.node], {}};
  for (const auto& atom : event.fiber)
    result.sources.push_back({atom.kind == 0 ? g.layout->input[atom.source] : g.layout->edge_target[atom.source], &atom,
                              atom.kind == 0 ? m.input_scale[atom.source] : m.agg_scale[atom.source]});
  if (result.sources.empty()) throw std::invalid_argument("Aggregate requires a nonempty fiber");
  return result;
}
void validate(const AggregateResult& result, const AggregateInput& input) {
  const auto& reference = input.sources[0].atom->value;
  auto tensor = [&](const Tensor& value) {
    if (!value.defined() || value.sizes() != reference.sizes() || value.device() != reference.device()
        || value.scalar_type() != reference.scalar_type()) throw std::invalid_argument("Aggregate returned incompatible tensor metadata");
  };
  tensor(result.value);
  std::set<Index> present;
  for (const auto& source : input.sources) present.insert(source.slot);
  Index previous = -1;
  for (const auto& source : result.contributions) {
    if (source.slot <= previous || !present.count(source.slot)) throw std::invalid_argument("Aggregate returned invalid contribution slots");
    tensor(source.value); previous = source.slot;
  }
}
AggregateResult bind(const AggregateResult& packed, const AggregateResult& ref) {
  if (packed.contributions.size() != ref.contributions.size()) throw std::invalid_argument("Aggregate batch changed contribution presence");
  AggregateResult result{semantic_value(packed.value, ref.value), {}};
  for (size_t j = 0; j < packed.contributions.size(); ++j) {
    if (packed.contributions[j].slot != ref.contributions[j].slot) throw std::invalid_argument("Aggregate batch changed contribution presence");
    auto source = ref.contributions[j];
    result.contributions.push_back({source.slot, semantic_value(packed.contributions[j].value,
      source.value.is_same(ref.value) ? result.value : source.value)});
  }
  return result;
}
}  // namespace
void evaluate_aggregate(const Graph& g, const Model& m, std::vector<Event>& events, const std::vector<size_t>& ids, bool packed) {
  if (ids.empty()) return;
  const auto& w = m.nodes[events[ids[0]].node];
  std::vector<AggregateInput> requests;
  for (auto i : ids) requests.push_back(request(g, m, events[i]));
  std::vector<AggregateResult> results;
  if (packed) {
    { at::NoGradGuard guard; results = w.aggregate_kernel->batch(w, requests); }
    if (results.size() != requests.size()) throw std::invalid_argument("Aggregate batch changed event count");
    for (size_t i = 0; i < requests.size(); ++i) {
      validate(results[i], requests[i]);
      if (at::GradMode::is_enabled()) {
        auto ref = w.aggregate_kernel->step(w, requests[i]); validate(ref, requests[i]);
        results[i] = bind(results[i], ref);
      }
    }
  } else {
    for (const auto& input : requests) results.push_back(w.aggregate_kernel->step(w, input));
  }
  for (size_t j = 0; j < ids.size(); ++j) {
    validate(results[j], requests[j]);
    auto& event = events[ids[j]];
    event.content = results[j].value; event.contributions = std::move(results[j].contributions);
  }
}
}  // namespace tide
