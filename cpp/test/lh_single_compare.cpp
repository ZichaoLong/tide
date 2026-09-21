#include "lh_single_graph.h"
#include <algorithm>

namespace lh_iocortex {
namespace {
void tensor(const at::Tensor& a, const at::Tensor& b) {
  require(a.defined() == b.defined(), "single graph tensor presence mismatch");
  if (a.defined()) close(a, b, "single graph tensor value mismatch");
}
void state(const tide::State& a, const tide::State& b) {
  require(a.last_time == b.last_time && a.observations == b.observations && a.slots.size() == b.slots.size(),
          "single graph state metadata mismatch");
  tensor(a.value, b.value); for (const auto& [name, t] : a.slots) tensor(t, b.slots.at(name));
}
void history(const tide::History& a, const tide::History& b) {
  require(a.last_time == b.last_time && a.scalars == b.scalars && a.node_maps == b.node_maps
          && a.tensors.size() == b.tensors.size(), "single graph history mismatch");
  for (const auto& [name, t] : a.tensors) tensor(t, b.tensors.at(name));
}
void atom(const tide::Atom& a, const tide::Atom& b) {
  require(a.key() == b.key(), "single graph atom identity mismatch"); tensor(a.value, b.value);
}
void atoms(std::vector<tide::Atom> a, std::vector<tide::Atom> b) {
  require(a.size() == b.size(), "single graph atom count mismatch");
  auto less = [](const auto& x, const auto& y) { return x.key() < y.key(); };
  std::sort(a.begin(), a.end(), less); std::sort(b.begin(), b.end(), less);
  for (size_t i = 0; i < a.size(); ++i) atom(a[i], b[i]);
}
void slots(const std::vector<tide::SlotValue>& a, const std::vector<tide::SlotValue>& b) {
  require(a.size() == b.size(), "single graph slot count mismatch");
  for (size_t i = 0; i < a.size(); ++i) {
    require(a[i].slot == b[i].slot, "single graph slot identity mismatch"); tensor(a[i].value, b[i].value);
  }
}
}  // namespace
void compare_result(const tide::Result& a, const tide::Result& b) {
  compare_continuation(a.continuation, b.continuation);
  require(a.trace.size() == b.trace.size() && a.outputs.size() == b.outputs.size(), "single graph result count mismatch");
  std::map<std::tuple<Index, Index, Index>, const tide::Event*> events;
  for (const auto& e : b.trace)
    require(events.emplace(std::make_tuple(e.time, e.batch, e.node), &e).second, "duplicate comparison event");
  for (const auto& e : a.trace) {
    auto it = events.find({e.time, e.batch, e.node}); require(it != events.end(), "single graph missing event");
    const auto& f = *it->second;
    require(e.active == f.active, "single graph route mismatch");
    atoms(e.fiber, f.fiber); tensor(e.content, f.content); tensor(e.proposal, f.proposal);
    tensor(e.descriptor, f.descriptor); tensor(e.control, f.control); tensor(e.comparison, f.comparison);
    tensor(e.next, f.next); tensor(e.full, f.full); state(e.old, f.old);
    state(e.proposed_state, f.proposed_state); state(e.comparison_state, f.comparison_state);
    state(e.next_state, f.next_state); history(e.history, f.history);
    slots(e.emitted, f.emitted); slots(e.contributions, f.contributions);
    require(e.sources.size() == f.sources.size(), "single graph source count mismatch");
    for (size_t i = 0; i < e.sources.size(); ++i) {
      require(e.sources[i].slot == f.sources[i].slot, "single graph logical source mismatch");
      atom(e.sources[i].atom, f.sources[i].atom); tensor(e.sources[i].scale, f.sources[i].scale);
    }
    events.erase(it);
  }
  auto outputs = [](const std::vector<tide::Output>& xs) {
    std::map<std::tuple<Index, Index, Index>, at::Tensor> result;
    for (const auto& x : xs)
      require(result.emplace(std::make_tuple(x.batch, x.time, x.port), x.value).second, "duplicate comparison output");
    return result;
  };
  auto remaining = outputs(b.outputs);
  for (const auto& [key, value] : outputs(a.outputs)) {
    auto it = remaining.find(key); require(it != remaining.end(), "single graph output identity mismatch");
    tensor(value, it->second); remaining.erase(it);
  }
  require(remaining.empty() && events.empty(), "single graph omitted result"); atoms(a.messages, b.messages);
}
}  // namespace lh_iocortex
