#pragma once
// Opt-in eager autograd metadata. No Python dependency; register before backward.
// Private graphs must register their own roots. Saved tensors are EXPLICIT ONLY:
// Node has no generic C++ accessor for every ATen operation's SavedVariables.
// Counters are cumulative observations, not execution time or allocator peaks.
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/function_hook.h>
#include <torch/csrc/autograd/variable.h>
#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace tide::autograd_census {
using Count = std::int64_t;
using Metrics = std::map<std::string, Count>;
using Node = torch::autograd::Node;
using Tensors = torch::autograd::variable_list;
struct NodeRecord {
  std::weak_ptr<Node> weak;
  std::string type;
  Metrics edges;
  Count null_edges = 0, executions = 0;
};
struct StorageRecord {
  c10::weak_intrusive_ptr<c10::StorageImpl> weak;
  Count bytes;
};
struct State {
  std::mutex mutex;
  std::unordered_map<Node*, size_t> identities;
  std::vector<NodeRecord> nodes;
  std::map<std::string, std::set<size_t>> groups;
  std::map<std::string, Count> registrations, roots, nondifferentiable;
  std::map<const c10::StorageImpl*, StorageRecord> storage;
  std::map<const c10::TensorImpl*, c10::weak_intrusive_ptr<c10::TensorImpl, c10::UndefinedTensorImpl>> tensors;
  Count explicit_saved_occurrences = 0, explicit_saved_bytes = 0, without_storage = 0;
};
inline std::atomic<bool> active{false};
inline std::mutex owner_mutex;
inline std::shared_ptr<State> owner;
inline bool enabled() noexcept { return active.load(std::memory_order_relaxed); }
inline std::shared_ptr<State> current() {
  if (!enabled()) return {};
  std::lock_guard<std::mutex> lock(owner_mutex);
  return owner;
}
// Reset/start only while measured workers are quiescent. Old hooks hold weak
// counters only and cannot keep any node, parameter, tensor or storage alive.
inline void reset(bool enabled) {
  active.store(false, std::memory_order_relaxed);
  std::lock_guard<std::mutex> lock(owner_mutex);
  owner = enabled ? std::make_shared<State>() : nullptr;
  active.store(enabled, std::memory_order_relaxed);
}
class Execution final : public torch::autograd::FunctionPreHook {
  std::weak_ptr<State> state_;
  size_t id_;
 public:
  Execution(const std::shared_ptr<State>& state, size_t id) : state_(state), id_(id) {}
  Tensors operator()(const Tensors& bars) override {
    if (auto state = state_.lock()) {
      std::lock_guard<std::mutex> lock(state->mutex);
      ++state->nodes[id_].executions;
    }
    return bars;
  }
};
inline void graph(const std::string& label, const Tensors& roots) {
  auto state = current();
  if (!state) return;
  if (label.empty()) throw std::invalid_argument("autograd census label must not be empty");
  std::lock_guard<std::mutex> lock(state->mutex);
  ++state->registrations[label]; state->roots[label] += roots.size();
  std::vector<std::shared_ptr<Node>> todo;
  for (const auto& t : roots) {
    if (t.defined() && t.requires_grad()) {
      auto edge = torch::autograd::impl::gradient_edge(t);
      if (edge.function) todo.push_back(std::move(edge.function));
    } else ++state->nondifferentiable[label];
  }
  std::set<Node*> seen;
  auto& group = state->groups[label];
  while (!todo.empty()) {
    auto node = std::move(todo.back()); todo.pop_back();
    if (!seen.insert(node.get()).second) continue;
    auto found = state->identities.find(node.get());
    size_t id;
    if (found == state->identities.end() || state->nodes[found->second].weak.expired()) {
      id = state->nodes.size();
      NodeRecord record; record.weak = node; record.type = node->name();
      for (const auto& edge : node->next_edges()) {
        if (edge.function) ++record.edges[edge.function->name()]; else ++record.null_edges;
      }
      state->nodes.push_back(std::move(record)); state->identities[node.get()] = id;
      node->add_pre_hook(std::make_unique<Execution>(state, id));
    } else id = found->second;
    group.insert(id);
    for (const auto& edge : node->next_edges()) if (edge.function) todo.push_back(edge.function);
  }
}
// Call once at an actual save boundary with the tensors being saved. This
// counts explicit save occurrences, including repeated aliases. It does not
// inspect ATen's internal saves or infer that their coverage is complete.
inline void saved(const Tensors& values) {
  auto state = current();
  if (!state) return;
  std::lock_guard<std::mutex> lock(state->mutex);
  for (const auto& t : values) {
    if (!t.defined()) continue;
    ++state->explicit_saved_occurrences;
    state->explicit_saved_bytes += t.numel()*t.element_size();
    const auto* impl = t.unsafeGetTensorImpl();
    if (!state->tensors.count(impl)) state->tensors.emplace(impl, t.getIntrusivePtr());
    if (t.layout() != at::kStrided) { ++state->without_storage; continue; }
    auto storage = t.storage(); const auto* key = storage.unsafeGetStorageImpl();
    if (!state->storage.count(key))
      state->storage.emplace(key, StorageRecord{storage.getWeakStorageImpl(), static_cast<Count>(storage.nbytes())});
  }
}
inline void counts(Metrics& result, const State& state, const std::string& prefix,
                   const std::set<size_t>& ids) {
  result[prefix+"unique_nodes"] = ids.size();
  result[prefix+"connected_edges"] = 0;
  result[prefix+"null_edges"] = 0;
  result[prefix+"node_executions"] = 0;
  for (auto id : ids) {
    const auto& node = state.nodes[id];
    ++result[prefix+"node_types/"+node.type];
    result[prefix+"null_edges"] += node.null_edges;
    result[prefix+"node_executions"] += node.executions;
    result[prefix+"execution_types/"+node.type] += node.executions;
    for (const auto& [type, count] : node.edges) {
      result[prefix+"connected_edges"] += count;
      result[prefix+"edge_types/"+node.type+" -> "+type] += count;
    }
  }
}
inline Metrics metrics() {
  auto state = current();
  if (!state) return {};
  std::lock_guard<std::mutex> lock(state->mutex);
  Metrics result{{"schema_version", 1}, {"coverage/saved_tensors_explicit_only", 1}};
  std::set<size_t> all;
  for (const auto& [label, ids] : state->groups) {
    const auto prefix = "graphs/"+label+"/";
    counts(result, *state, prefix, ids); all.insert(ids.begin(), ids.end());
    result[prefix+"registrations"] = state->registrations[label];
    result[prefix+"roots"] = state->roots[label];
    result[prefix+"nondifferentiable_roots"] = state->nondifferentiable[label];
  }
  counts(result, *state, "total/", all);
  result["explicit_saved/occurrences"] = state->explicit_saved_occurrences;
  result["explicit_saved/logical_bytes"] = state->explicit_saved_bytes;
  result["explicit_saved/unique_tensor_impls"] = state->tensors.size();
  result["explicit_saved/unique_storages"] = state->storage.size();
  result["explicit_saved/without_storage"] = state->without_storage;
  result["explicit_saved/unique_storage_bytes"] = 0;
  result["explicit_saved/storage_bytes_still_alive"] = 0;
  for (const auto& [ptr, record] : state->storage) {
    result["explicit_saved/unique_storage_bytes"] += record.bytes;
    if (!record.weak.expired()) result["explicit_saved/storage_bytes_still_alive"] += record.bytes;
  }
  return result;
}
} // namespace tide::autograd_census
