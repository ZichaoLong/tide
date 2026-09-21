#include "tide/ops.h"
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
  for (const auto& w : m.nodes) {
    check_tensor(w.decay, ref, {d}); check_tensor(w.weight, ref, {d, d});
    check_tensor(w.bias, ref, {d}); check_tensor(w.read, ref, {d});
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
  }
  for (const auto& [owner, counts] : q.history) {
    require(batch(owner.first) && owner.second >= 0 && owner.second < r, "invalid history owner");
    for (const auto& [v, count] : counts)
      require(v >= 0 && v < n && g.nodes[v].region == owner.second && count >= 0, "invalid selector history");
  }
  for (const auto& [owner, last] : q.ledger)
    require(batch(owner.first) && owner.second >= 0 && owner.second < p && last.first >= 0
            && last.second >= 0 && last.second < q.cut, "invalid input ledger");
  auto sorted = external;
  std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
    return std::tie(a.batch, a.port, a.position) < std::tie(b.batch, b.port, b.position);
  });
  std::vector<Atom> atoms;
  for (const auto& x : sorted) {
    require(batch(x.batch) && x.port >= 0 && x.port < p && x.position >= 0
            && x.time >= q.cut && x.time < stop, "invalid external coordinate");
    const Owner owner{x.batch, x.port};
    auto it = q.ledger.find(owner);
    const auto last = it == q.ledger.end() ? Owner{-1, -1} : it->second;
    require(x.position == last.first + 1 && x.time > last.second, "noncontiguous/nonmonotonic port history");
    check_tensor(x.value, ref, {m.width()});
    q.ledger[owner] = {x.position, x.time};
    atoms.push_back({x.batch, g.inputs[x.port], x.time, 0, x.port, x.position, x.value});
  }
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
  return atoms;
}
}  // namespace tide
