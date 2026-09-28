#include "placement.h"
#include <c10/core/DeviceGuard.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <algorithm>
#include <set>
#include <stdexcept>
#if PORTABLE_TORCH_ENABLE_NPU
#include <torch_npu/torch_npu.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#endif

namespace accelerator_scale {
void finalize() {
#if PORTABLE_TORCH_ENABLE_NPU
  torch_npu::finalize_npu();
#endif
}
Placement place(pdg_scale::Fixture& f, at::Device first, Index count, const std::string& policy, bool resident) {
  at::NoGradGuard guard;
  if (count < 1 || count > 16 || (first.is_cpu() && count != 1))
    throw std::invalid_argument("placement requires 1..16 devices; CPU supports one");
  if (!first.is_cpu()) {
    c10::impl::VirtualGuardImpl api(first.type());
    if (first.index()+count > api.deviceCount()) throw std::invalid_argument("requested devices unavailable");
  }
  Placement result; result.policy = policy; result.resident = resident;
  std::vector<std::shared_ptr<std::mutex>> locks;
  for (Index i = 0; i < count; ++i) {
    result.devices.push_back(first.is_cpu() ? first : at::Device(first.type(), first.index()+i));
    result.parameter_bytes.push_back(0); locks.push_back(std::make_shared<std::mutex>());
  }
  std::map<std::pair<const c10::TensorImpl*, Index>, Tensor> copies;
  std::map<const c10::TensorImpl*, std::pair<Index, Tensor>> parameters;
  std::set<const c10::TensorImpl*> owners;
  std::map<const c10::StorageImpl*, Tensor> storage_owners;
  for (const auto& p : f.owners) {
    owners.insert(p.unsafeGetTensorImpl()); storage_owners.emplace(p.storage().unsafeGetStorageImpl(), p);
  }
  std::function<Tensor(const Tensor&, Index)> copy = [&](const Tensor& value, Index slot) -> Tensor {
    const auto key = std::make_pair(value.unsafeGetTensorImpl(), slot);
    if (copies.count(key)) return copies.at(key);
    const auto base = storage_owners.find(value.storage().unsafeGetStorageImpl());
    if (!owners.count(key.first) && base != storage_owners.end()) {
      // Reconstruct scalar coefficient views from their one shared owner.
      auto target = copy(base->second, slot);
      at::AutoGradMode track(true);
      auto view = target.as_strided(value.sizes(), value.strides(),
        value.storage_offset()-base->second.storage_offset()+target.storage_offset());
      copies.emplace(key, view); return view;
    }
    if (owners.count(key.first) && parameters.count(key.first) && parameters.at(key.first).first != slot)
      throw std::invalid_argument("a shared parameter cannot be duplicated across placement shards");
    const auto device = result.devices.at(slot);
    c10::DeviceGuard device_guard(device);
    auto target = device.is_cpu() ? value : value.detach().to(device).set_requires_grad(value.requires_grad());
    copies.emplace(key, target);
    if (owners.count(key.first)) {
      parameters[key.first] = {slot, target};
      result.parameter_bytes[slot] += value.numel()*value.element_size();
    }
    return target;
  };
  std::vector<int64_t> node_bytes;
  for (size_t n = 0; n < f.model.nodes.size(); ++n) {
    int64_t bytes = 0;
    for (const auto& [name, p] : f.model.nodes[n].extra)
      if (owners.count(p.unsafeGetTensorImpl())) bytes += p.numel()*p.element_size();
    node_bytes.push_back(bytes);
  }
  const auto plan = partition(f.graph, node_bytes, count, policy);
  result.node_device = plan.shards; result.node_load_limit = plan.limit;
  result.edges = f.graph.edges.size();
  for (const auto& edge : f.graph.edges)
    result.cut_edges += plan.shards[edge.source] != plan.shards[edge.target];
  for (size_t n = 0; n < f.model.nodes.size(); ++n) {
    const Index slot = plan.shards[n];
    auto& host_weights = f.model.nodes[n];
    auto node = std::make_shared<DeviceNode>(); node->weights = host_weights; node->mutex = locks[slot];
    auto& w = node->weights;
    w.full_kind = f.graph.nodes[n].full;
    if (!w.full_kernel) w.full_kernel = make_full_kernel(f.graph.nodes[n]);
    w.decay = copy(w.decay, slot); w.weight = copy(w.weight, slot);
    w.bias = copy(w.bias, slot); w.read = copy(w.read, slot);
    for (auto& [name, value] : w.extra)
      if (resident || name.rfind("agg_logit_", 0) != 0) value = copy(value, slot);
    if (resident) { host_weights = std::move(w); continue; }
    host_weights.kernel = state_adapter(node); host_weights.full_kernel = full_adapter(node);
    // All-softmax Aggregate and its small coefficient owners stay on the CPU.
    for (auto it = host_weights.extra.begin(); it != host_weights.extra.end();)
      if (it->first.rfind("agg_logit_", 0) == 0) ++it;
      else it = host_weights.extra.erase(it);
  }
  auto lightest = [&] { return std::min_element(result.parameter_bytes.begin(), result.parameter_bytes.end())-result.parameter_bytes.begin(); };
  f.embedding = copy(f.embedding, lightest()); f.head = copy(f.head, lightest());
  if (resident) {
    for (size_t i = 0; i < f.graph.inputs.size(); ++i)
      f.model.input_scale[i] = copy(f.model.input_scale[i], plan.shards[f.graph.inputs[i]]);
    for (size_t i = 0; i < f.graph.outputs.size(); ++i)
      f.model.output_scale[i] = copy(f.model.output_scale[i], plan.shards[f.graph.outputs[i]]);
    for (size_t i = 0; i < f.graph.edges.size(); ++i) {
      f.model.agg_scale[i] = copy(f.model.agg_scale[i], plan.shards[f.graph.edges[i].target]);
      f.model.edge_scale[i] = copy(f.model.edge_scale[i], plan.shards[f.graph.edges[i].source]);
    }
  }
  for (auto& owner : f.owners) {
    auto it = parameters.find(owner.unsafeGetTensorImpl());
    if (it != parameters.end()) owner = it->second.second;
  }
  synchronize(result);
  return result;
}
void synchronize(const Placement& placement) {
  for (auto device : placement.devices) portable_torch::synchronize(device);
}
void reset_memory(const Placement& placement) {
#if PORTABLE_TORCH_ENABLE_NPU
  for (auto device : placement.devices) if (device.type() == c10::DeviceType::PrivateUse1)
    c10_npu::NPUCachingAllocator::resetPeakStats(device.index());
#endif
}
std::map<std::string, double> memory(const Placement& placement) {
  std::map<std::string, double> result{{"placement/cut_edges", double(placement.cut_edges)},
    {"placement/cut_fraction", placement.edges ? double(placement.cut_edges)/placement.edges : 0.},
    {"placement/node_load_limit_bytes", double(placement.node_load_limit)}};
  for (size_t i = 0; i < placement.devices.size(); ++i) {
    const auto prefix = "memory/device"+std::to_string(i)+"/";
    result[prefix+"parameter_bytes"] = placement.parameter_bytes[i];
#if PORTABLE_TORCH_ENABLE_NPU
    auto device = placement.devices[i];
    if (device.type() == c10::DeviceType::PrivateUse1) {
      auto stats = c10_npu::NPUCachingAllocator::getDeviceStats(device.index());
      result[prefix+"allocated_bytes"] = stats.allocated_bytes[0].current;
      result[prefix+"peak_allocated_bytes"] = stats.allocated_bytes[0].peak;
      result[prefix+"reserved_bytes"] = stats.reserved_bytes[0].current;
      result[prefix+"peak_reserved_bytes"] = stats.reserved_bytes[0].peak;
    }
#endif
  }
  return result;
}
}  // namespace accelerator_scale
