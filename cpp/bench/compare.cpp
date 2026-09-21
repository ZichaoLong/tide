#include "streaming.h"
#include <stdexcept>

namespace tide_bench {
namespace {
void require(bool ok, const char* what) {
  if (!ok) throw std::runtime_error(std::string("benchmark parity: ")+what);
}
void tensor(const at::Tensor& a, const at::Tensor& b) {
  require(a.defined() == b.defined(), "tensor presence");
  if (!a.defined()) return;
  require(a.scalar_type() == b.scalar_type() && a.sizes() == b.sizes(), "tensor metadata");
  const bool fp64 = a.scalar_type() == at::kDouble;
  require(at::allclose(a, b, fp64 ? 1e-8 : 1e-5, fp64 ? 1e-10 : 1e-6), "tensor value");
}
void slots(const std::map<std::string, at::Tensor>& a, const std::map<std::string, at::Tensor>& b) {
  require(a.size() == b.size(), "tensor slots");
  for (const auto& [key, value] : a) tensor(value, b.at(key));
}
void state(const tide::State& a, const tide::State& b) {
  require(a.last_time == b.last_time && a.observations == b.observations, "state clock/count");
  tensor(a.value, b.value); slots(a.slots, b.slots);
}
void history(const tide::History& a, const tide::History& b) {
  require(a.last_time == b.last_time && a.scalars == b.scalars && a.node_maps == b.node_maps, "history");
  slots(a.tensors, b.tensors);
}
void atom(const tide::Atom& a, const tide::Atom& b) {
  require(a.key() == b.key(), "atom identity"); tensor(a.value, b.value);
}
void atoms(const std::vector<tide::Atom>& a, const std::vector<tide::Atom>& b) {
  require(a.size() == b.size(), "atom count");
  for (size_t i = 0; i < a.size(); ++i) atom(a[i], b[i]);
}
void emitted(const std::vector<tide::SlotValue>& a, const std::vector<tide::SlotValue>& b) {
  require(a.size() == b.size(), "slot count");
  for (size_t i = 0; i < a.size(); ++i) {
    require(a[i].slot == b[i].slot, "slot identity"); tensor(a[i].value, b[i].value);
  }
}
}
void compare(const tide::Result& a, const tide::Result& b, bool traces) {
  const auto& x = a.continuation; const auto& y = b.continuation;
  // The anchor contains only the active rings. All their IDs and edges form an
  // unchanged prefix of the dormant extension; only whole-graph identity differs.
  require(!x.identity.empty() && !y.identity.empty(), "graph identity presence");
  require(x.cut == y.cut && x.batch_size == y.batch_size && x.ledger == y.ledger, "cut/batch/ledger");
  require(x.states.size() == y.states.size() && x.history.size() == y.history.size(), "state/history owners");
  for (const auto& [owner, value] : x.states) state(value, y.states.at(owner));
  for (const auto& [owner, value] : x.history) history(value, y.history.at(owner));
  atoms(x.pending, y.pending);
  require(a.outputs.size() == b.outputs.size(), "output count");
  for (size_t i = 0; i < a.outputs.size(); ++i) {
    const auto& u = a.outputs[i]; const auto& v = b.outputs[i];
    require(std::tie(u.batch, u.time, u.port) == std::tie(v.batch, v.time, v.port), "output identity");
    tensor(u.value, v.value);
  }
  if (!traces) return;
  atoms(a.messages, b.messages); require(a.trace.size() == b.trace.size(), "trace count");
  for (size_t i = 0; i < a.trace.size(); ++i) {
    const auto& u = a.trace[i]; const auto& v = b.trace[i];
    require(std::tie(u.batch, u.node, u.time, u.active) == std::tie(v.batch, v.node, v.time, v.active), "event/route");
    atoms(u.fiber, v.fiber); history(u.history, v.history);
    state(u.old, v.old); state(u.proposed_state, v.proposed_state);
    state(u.comparison_state, v.comparison_state); state(u.next_state, v.next_state);
    tensor(u.content, v.content); tensor(u.proposal, v.proposal); tensor(u.descriptor, v.descriptor);
    tensor(u.control, v.control); tensor(u.comparison, v.comparison); tensor(u.next, v.next); tensor(u.full, v.full);
    emitted(u.emitted, v.emitted); emitted(u.contributions, v.contributions);
    require(u.sources.size() == v.sources.size(), "source count");
    for (size_t j = 0; j < u.sources.size(); ++j) {
      require(u.sources[j].slot == v.sources[j].slot, "source slot");
      atom(u.sources[j].atom, v.sources[j].atom); tensor(u.sources[j].scale, v.sources[j].scale);
    }
  }
}
}  // namespace tide_bench
