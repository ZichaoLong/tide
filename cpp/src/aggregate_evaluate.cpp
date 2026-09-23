#include "tide/operator_profile.h"
#include "tide/aggregate.h"
#include "tide/autograd.h"
#include <ATen/core/grad_mode.h>
#include <algorithm>
#include <set>
#include <stdexcept>

namespace tide {
namespace {
AggregateInput request(const Graph& g, const Model& m, const Event& event) {
  AggregateInput result{event.time, g.source_counts[event.node], {}};
  std::set<Index> present;
  for (const auto& atom : event.fiber) {
    auto visible = atom;
    if (atom.kind == 1 && !g.origins.empty() && g.origin_index[atom.source] != -1) {
      const auto& origin = g.origins[g.origin_index[atom.source]];
      if (atom.position % origin.stride) throw std::invalid_argument("message does not lie on input origin clock");
      visible.kind = 0; visible.source = origin.port; visible.position /= origin.stride;
    }
    const auto slot = atom.kind == 0 ? g.source_domain->input[atom.source] : g.source_domain->edge_target[atom.source];
    if (!present.insert(slot).second) throw std::invalid_argument("duplicate logical source in complete fiber");
    result.sources.push_back({slot, visible,
                              atom.kind == 0 ? m.input_scale[atom.source] : m.agg_scale[atom.source]});
  }
  if (result.sources.empty()) throw std::invalid_argument("Aggregate requires a nonempty fiber");
  if (!g.origins.empty())
    std::sort(result.sources.begin(), result.sources.end(), [](const auto& a, const auto& b) { return a.atom.key() < b.atom.key(); });
  return result;
}
void validate(const AggregateResult& result, const AggregateInput& input) {
  const auto& reference = input.sources[0].atom.value;
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
void validate_transport(const AggregateBatch& batch, const std::vector<AggregateInput>& requests) {
  const auto& ref = requests.at(0).sources.at(0).atom.value;
  auto matrix = [&](const Tensor& value, Index rows) {
    if (!value.defined() || value.dim() != 2 || value.size(0) != rows || value.size(1) != ref.numel()
        || value.device() != ref.device() || value.scalar_type() != ref.scalar_type())
      throw std::invalid_argument("Aggregate transport changed tensor metadata");
  };
  if (batch.contents.defined()) matrix(batch.contents, requests.size());
  if (!batch.sources) return;
  const auto& s = *batch.sources;
  if (s.offsets.size() != requests.size()+1 || s.offsets.front() != 0)
    throw std::invalid_argument("Aggregate transport changed event offsets");
  Index count = 0;
  for (size_t i = 0; i < requests.size(); ++i) {
    for (const auto& source : requests[i].sources) {
      if (count >= static_cast<Index>(s.slots.size()) || s.slots[count++] != source.slot)
        throw std::invalid_argument("Aggregate transport changed source slots");
    }
    if (s.offsets[i+1] != count) throw std::invalid_argument("Aggregate transport changed event offsets");
  }
  if (count != static_cast<Index>(s.slots.size())) throw std::invalid_argument("Aggregate transport changed source count");
  matrix(s.values, count);
}
}  // namespace
Tensor evaluate_aggregate(const Graph& g, const Model& m, std::vector<Event>& events, const std::vector<size_t>& ids,
                          bool packed, bool packed_sources) {
  if (ids.empty()) return {};
  op_profile::Scope profile(op_profile::Aggregate);
  const auto& w = m.nodes[events[ids[0]].node];
  std::vector<AggregateInput> requests;
  for (auto i : ids) requests.push_back(request(g, m, events[i]));
  std::vector<AggregateResult> results;
  AggregateBatch transport;
  if (packed) {
    { at::NoGradGuard guard;
      if (packed_sources) {
        transport = w.aggregate_kernel->source_batch(w, requests); validate_transport(transport, requests);
        results = std::move(transport.events);
      } else results = w.aggregate_kernel->batch(w, requests);
    }
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
    event.sources = std::move(requests[j].sources);
    event.source_batch = transport.sources; event.source_row = transport.sources ? static_cast<Index>(j) : -1;
  }
  return transport.contents;
}
}  // namespace tide
