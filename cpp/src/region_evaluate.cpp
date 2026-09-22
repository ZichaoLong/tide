#include "tide/region.h"
#include <algorithm>
#include <stdexcept>

namespace tide {
namespace {
void check_tensor(const Tensor& value, const Tensor& ref, const char* message) {
  if (!value.defined() || value.device() != ref.device() || value.scalar_type() != ref.scalar_type()
      || !at::isfinite(value).all().item<bool>()) throw std::invalid_argument(message);
}
}  // namespace
void validate_history(const History& h, const RegionLayout& layout, const Tensor& ref, Index time) {
  if (h.last_time < -1 || h.last_time > time) throw std::invalid_argument("invalid region history clock");
  for (const auto& [name, value] : h.scalars)
    if (name.empty()) throw std::invalid_argument("invalid region history field names");
  for (const auto& [name, counts] : h.node_maps) {
    if (name.empty()) throw std::invalid_argument("invalid region history field names");
    for (const auto& [node, count] : counts) layout.slot(node);
  }
  for (const auto& [name, value] : h.tensors) {
    if (name.empty()) throw std::invalid_argument("invalid region history field names");
    check_tensor(value, ref, "incompatible or nonfinite region history tensor");
  }
}
Selection evaluate_selection(const Graph& g, const Model& m, const History* previous_history,
                             const std::vector<Event>& events, const std::vector<size_t>& ids) {
  if (ids.empty()) throw std::invalid_argument("empty region evaluation");
  const auto& first = events[ids[0]];
  const Owner owner{first.batch, g.nodes[first.node].region};
  const auto layout = region_layout(g, owner.second);
  const auto& w = m.regions[owner.second];
  const auto& ref = m.nodes[0].bias;
  std::vector<Candidate> candidates;
  std::set<Index> nodes;
  Index previous = -1;
  for (auto i : ids) {
    const auto& e = events[i];
    if (e.batch != first.batch || e.time != first.time || e.node <= previous || g.nodes[e.node].region != owner.second)
      throw std::invalid_argument("invalid region candidate domain/order");
    candidates.push_back({e.node, e.descriptor}); nodes.insert(e.node); previous = e.node;
  }
  History initial;
  if (!previous_history) {
    initial = w.kernel->initial(w, layout, ref);
    validate_history(initial, layout, ref, first.time-1); w.kernel->validate_history(initial, layout);
  }
  const auto& old = previous_history ? *previous_history : initial;
  auto result = w.kernel->step(w, {old, first.time, candidates, layout, ref.options()});
  if (static_cast<Index>(result.active.size()) > layout.spec.budget
      || !std::includes(nodes.begin(), nodes.end(), result.active.begin(), result.active.end()))
    throw std::invalid_argument("region selector returned invalid active subset/capacity");
  if (result.controls.size() != nodes.size()) throw std::invalid_argument("region selector returned invalid control domain");
  for (const auto& [node, control] : result.controls) {
    if (!nodes.count(node)) throw std::invalid_argument("region selector returned invalid control domain");
    check_tensor(control, ref, "region selector returned incompatible or nonfinite control");
  }
  validate_history(result.history, layout, ref, first.time); w.kernel->validate_history(result.history, layout);
  return result;
}
void commit_selection(Continuation& q, Owner owner, Selection result, std::vector<Event>& events,
                      const std::vector<size_t>& ids, bool trace) {
  q.history[owner] = std::move(result.history);
  for (auto i : ids) {
    auto& e = events[i]; e.active = result.active.count(e.node); e.control = std::move(result.controls.at(e.node));
    if (trace) e.history = q.history.at(owner);
  }
}
void select_events(const Graph& g, const Model& m, Continuation& q, std::vector<Event>& events,
                   const std::vector<size_t>& ids, bool trace) {
  if (ids.empty()) return;
  const auto& first = events[ids[0]];
  const Owner owner{first.batch, g.nodes[first.node].region};
  const auto it = q.history.find(owner);
  auto result = evaluate_selection(g, m, it == q.history.end() ? nullptr : &it->second, events, ids);
  commit_selection(q, owner, std::move(result), events, ids, trace);
}
}  // namespace tide
