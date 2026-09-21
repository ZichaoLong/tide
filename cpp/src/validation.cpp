#include "tide/ops.h"
#include "tide/kernel.h"
#include "tide/full.h"
#include "tide/aggregate.h"
#include "tide/read.h"
#include "tide/next.h"
#include "tide/region.h"
#include <algorithm>
#include <set>
#include <stdexcept>

namespace tide {
namespace {
void require(bool ok, const char* message) { if (!ok) throw std::invalid_argument(message); }
void check_tensor(const Tensor& x, const Tensor& reference, at::IntArrayRef shape) {
  require(x.defined() && x.device().is_cpu() && x.scalar_type() == reference.scalar_type()
          && x.sizes() == shape, "incompatible tensor dtype/device/shape");
  require(at::isfinite(x).all().item<bool>(), "nonfinite tensor");
}
}  // namespace
void validate_model(const Graph& g, const Model& m) {
  require(m.nodes.size() == g.nodes.size() && !m.nodes.empty(), "node weight count mismatch");
  const auto& ref = m.nodes[0].bias;
  require(ref.defined() && ref.dim() == 1 && ref.numel() > 0, "invalid model width");
  require(ref.scalar_type() == at::kFloat || ref.scalar_type() == at::kDouble, "FP32/FP64 required");
  const Index d = ref.numel();
  for (size_t node = 0; node < m.nodes.size(); ++node) {
    const auto& w = m.nodes[node];
    check_tensor(w.decay, ref, {d}); check_tensor(w.weight, ref, {d, d});
    check_tensor(w.bias, ref, {d}); check_tensor(w.read, ref, {d});
    require(static_cast<bool>(w.kernel), "state kernel is not configured");
    w.kernel->validate_policy(g.nodes[node], g.source_counts[node]);
    w.kernel->validate_weights(w);
    require(static_cast<bool>(w.read_kernel), "Read kernel is not configured");
    w.read_kernel->validate_weights(w);
    require(static_cast<bool>(w.next_kernel), "Next kernel is not configured");
    w.next_kernel->validate_weights(w);
    for (const auto& [name, value] : w.extra) check_tensor(value, ref, value.sizes());
    require(static_cast<bool>(w.full_kernel), "Full kernel is not configured");
    w.full_kernel->validate_weights(w, g.outgoing_ports.offsets[node+1] - g.outgoing_ports.offsets[node]);
    require(static_cast<bool>(w.aggregate_kernel), "Aggregate kernel is not configured");
    w.aggregate_kernel->validate_weights(w, g.source_counts[node]);
  }
  require(m.regions.size() == g.regions.size(), "region weight count mismatch");
  for (size_t r = 0; r < m.regions.size(); ++r) {
    const auto& w = m.regions[r];
    require(static_cast<bool>(w.kernel), "region kernel is not configured");
    for (const auto& [name, value] : w.extra) check_tensor(value, ref, value.sizes());
    w.kernel->validate_weights(w, region_layout(g, r));
  }
  require(m.input_scale.size() == g.inputs.size() && m.agg_scale.size() == g.edges.size()
          && m.edge_scale.size() == g.edges.size() && m.output_scale.size() == g.outputs.size(),
          "source scale count mismatch");
  for (const auto* group : {&m.input_scale, &m.agg_scale, &m.edge_scale, &m.output_scale})
    for (const auto& w : *group) check_tensor(w, ref, {});
}
std::vector<Atom> validate_window(const Graph& g, const Model& m, Continuation& q,
                                 const std::vector<External>& external, Index stop, Index seal) {
  require(q.identity == g.identity && q.batch_size > 0, "continuation identity/batch mismatch");
  require(q.cut >= 0 && q.cut <= stop && stop <= seal, "window is unsealed or precedes cut");
  const auto& ref = m.nodes[0].bias;
  const auto n = static_cast<Index>(g.nodes.size());
  const auto r = static_cast<Index>(g.regions.size());
  const auto p = static_cast<Index>(g.inputs.size());
  auto batch = [&q](Index b) { return b >= 0 && b < q.batch_size; };
  for (const auto& [owner, state] : q.states) {
    require(batch(owner.first) && owner.second >= 0 && owner.second < n, "invalid state owner");
    require(state.last_time < q.cut && state.last_time >= -1 && state.observations >= 0, "invalid state clock");
    check_tensor(state.value, ref, {m.width()});
    m.nodes[owner.second].kernel->validate_state(m.nodes[owner.second], state);
    for (const auto& [name, value] : state.slots) check_tensor(value, ref, value.sizes());
  }
  for (const auto& [owner, history] : q.history) {
    require(batch(owner.first) && owner.second >= 0 && owner.second < r, "invalid history owner");
    auto layout = region_layout(g, owner.second);
    validate_history(history, layout, ref, q.cut-1);
    m.regions[owner.second].kernel->validate_history(history, layout);
  }
  for (const auto& [owner, last] : q.ledger)
    require(batch(owner.first) && owner.second >= 0 && owner.second < p && last.first >= 0
            && last.second >= 0 && last.second < q.cut, "invalid input ledger");
  auto input = validate_external(g, m, q, external, stop, seal);
  std::set<std::tuple<Index, Index, Index>> seen;
  for (const auto& a : q.pending) {
    require(a.kind == 1 && a.source >= 0 && a.source < static_cast<Index>(g.edges.size()), "invalid pending edge");
    const auto& e = g.edges[a.source];
    require(batch(a.batch) && a.node == e.target && a.position >= 0 && a.position < q.cut
            && a.time >= q.cut && a.time >= a.position && a.time - a.position == e.delay,
            "invalid pending coordinate");
    require(seen.insert({a.batch, a.source, a.position}).second, "duplicate pending message");
    check_tensor(a.value, ref, {m.width()});
  }
  for (const auto& [owner, last] : input.ledger_updates) q.ledger[owner] = last;
  return input.atoms;
}
}  // namespace tide
