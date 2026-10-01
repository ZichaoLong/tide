#pragma once
#include "placement.h"

namespace accelerator_scale::bounded {
// Finite predicated dataflow. Tensor masks own presence and decisions; C++ only
// builds the fixed topology/time expansion. No tensor scalar is read by run().
// The explicit first-order VJP also carries structural leaf connectivity: a
// numerically zero derivative does not decide whether a gradient is absent.
struct Value { Tensor data, dependencies; };
struct State {
  Value value;
  Tensor last, observations, seen, key, cache_value, bias, valid, cache_dependencies;
};
struct History { Tensor selected, affected, last, seen; };
struct Message {
  Index node, time, kind, source, position, slot;
  Value value;
  Tensor present;
};
struct Event {
  Index node, time;
  Tensor candidate, active;
  Value content, descriptor, control, fresh;
  State old, proposal, next;
  History history;
  std::vector<Message> fiber;
  std::vector<Value> contributions, emitted;
  std::vector<Index> emit_slots;
};
struct Window {
  std::vector<State> states;
  std::vector<History> histories;
  std::vector<Event> events;
  std::vector<Message> messages, pending;
  std::vector<Value> logits;
  std::vector<Tensor> output_present;
  // Compact diagnostic masks; reductions happen only at the completed window
  // boundary. They never supply a decision to the execution schedule.
  std::vector<Tensor> candidates, selected, edge_presence;
};
struct Limits {
  Index tokens = 3, batch = 2;
  int64_t max_workspace_bytes = int64_t(4) << 30;
  bool trace = true, connectivity = true;
};
// Public graph identity remains independent of this finite execution plan.
struct Schedule {
  Index period;
  std::vector<Index> edge_rows; // -1 broadcasts; otherwise the owner's projection row.
  int64_t workspace_bound;
};
// Topology-only refusal bound; callers can check it before allocating weights.
int64_t estimate_workspace(const pdg_scale::Topology&,const std::string& memory,
                          Index width,at::ScalarType,const Limits&,bool training);
class Program {
 public:
  Program(pdg_scale::Fixture&, const pdg_scale::Topology&, Placement, Limits);
  Program(pdg_scale::Fixture&, Schedule, Placement, Limits);
  Window run(const Tensor& ids, const std::vector<Tensor>& external_roots = {}) const;
  const std::vector<Tensor>& leaves() const { return leaves_; }
  const Graph& graph() const { return f_.graph; }
  const Limits& limits() const { return limits_; }
  Index period() const { return period_; }
  const std::vector<std::vector<Index>>& members() const { return members_; }
  int64_t workspace_bound() const { return workspace_bound_; }
  Value loss(const Window&, const Tensor& ids) const;
  std::vector<Tensor> vjp(const Value&, const Tensor& cotangent, bool retain = true,
                        const std::vector<Tensor>& external_roots = {}) const;
  Tensor dependency(const Tensor&, at::Device) const;
  Tensor empty_dependencies(at::Device) const;
  Value copy(const Value&, at::Device) const;
 private:
  pdg_scale::Fixture& f_;
  Index period_;
  Placement placement_;
  Limits limits_;
  int64_t workspace_bound_ = 0;
  std::vector<Tensor> leaves_;
  std::map<const c10::StorageImpl*, Index> owner_;
  std::vector<std::vector<Index>> members_;
  std::vector<Index> edge_rows_;
  std::map<std::string, Tensor> leaf_ids_;
  State initial(Index) const;
  Event update(Index, Index, const State&, std::vector<Message>) const;
  void select(Index, Index, std::map<Index, Event>&, History&) const;
  void finish(Event&) const;
};
// Export is an explicit window boundary; all scalar/index copies are here.
Result export_result(const Program&, const Window&);
Value row(const Value&, Index);
Value root(const Tensor&, const Tensor& dependencies);
std::vector<Tensor> export_gradients(const Value&, std::vector<Tensor>);
Tensor masked_rows(const Tensor&, const Tensor&);
Tensor choose_dependencies(const Tensor&, const Tensor&, const Tensor&);
Tensor move(const Tensor&, at::Device);
}  // namespace accelerator_scale::bounded
