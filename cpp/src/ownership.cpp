#include "tide/ownership.h"
#include "tide/region.h"
#include "tide/parameters.h"
#include <ATen/core/grad_mode.h>
#include <algorithm>
#include <stdexcept>

namespace tide {
std::vector<at::Device> model_devices(const Model& m) {
  std::vector<at::Device> result;
  for (const auto& p : m.parameters(false).owners())
    if (std::find(result.begin(), result.end(), p.value.device()) == result.end()) result.push_back(p.value.device());
  return result;
}
const Tensor& region_reference(const Graph& g, const Model& m, Index r) {
  const auto& w = m.regions.at(r);
  if (!w.extra.empty()) return w.extra.begin()->second;
  const auto members = region_layout(g, r).members;
  return m.nodes.at(members.empty() ? 0 : members[0]).bias;
}
Model place_payloads(const Graph& g, const Model& m, const std::vector<at::Device>& devices) {
  if (devices.size() != g.nodes.size() || m.nodes.size() != g.nodes.size())
    throw std::invalid_argument("payload owner map must cover every node");
  std::optional<c10::DeviceType> accelerator;
  for (const auto& d : devices) {
    if (d.is_cpu()) continue;
    if ((!d.is_cuda() && d.type() != c10::DeviceType::PrivateUse1) || !d.has_index())
      throw std::invalid_argument("payload owners require CPU or indexed CUDA/NPU devices");
    if (accelerator && *accelerator != d.type()) throw std::invalid_argument("payload owners cannot mix accelerator backends");
    accelerator = d.type();
  }
  std::map<const c10::TensorImpl*, at::Device> owners;
  auto assign = [&](const Tensor& t, at::Device d, bool required) {
    if (d.is_cpu()) d = at::Device(at::kCPU);
    auto found = owners.emplace(t.unsafeGetTensorImpl(), d);
    if (required && found.first->second != d)
      throw std::invalid_argument("shared node parameter requires co-located payload owners");
  };
  for (size_t i = 0; i < m.nodes.size(); ++i) {
    const auto& w = m.nodes[i];
    for (const auto& t : {w.decay,w.weight,w.bias,w.read}) assign(t, devices[i], true);
    for (const auto& [_,t] : w.extra) assign(t, devices[i], true);
  }
  for (size_t r = 0; r < m.regions.size(); ++r) {
    const auto members = region_layout(g,r).members;
    auto d = devices[members.empty() ? 0 : members[0]];
    for (const auto& [_,t] : m.regions[r].extra) {
      auto found = owners.find(t.unsafeGetTensorImpl());
      if (found != owners.end()) { d = found->second; break; }
    }
    for (const auto& [_,t] : m.regions[r].extra) assign(t, d, true);
  }
  for (size_t i=0;i<m.input_scale.size();++i) assign(m.input_scale[i], devices.at(g.inputs.at(i)), false);
  for (size_t i=0;i<m.output_scale.size();++i) assign(m.output_scale[i], devices.at(g.outputs.at(i)), false);
  for (size_t i=0;i<m.agg_scale.size();++i) assign(m.agg_scale[i], devices.at(g.edges.at(i).target), false);
  for (size_t i=0;i<m.edge_scale.size();++i) assign(m.edge_scale[i], devices.at(g.edges.at(i).source), false);
  Model result = m;
  std::map<const c10::TensorImpl*, Tensor> copied;
  auto move = [&](Tensor& t) {
    const auto id = t.unsafeGetTensorImpl();
    auto found = copied.find(id);
    if (found == copied.end()) {
      auto d = owners.at(id);
      auto value = t.device() == d ? t : t.detach().to(d).set_requires_grad(t.requires_grad());
      found = copied.emplace(id, value).first;
    }
    t = found->second;
  };
  for (auto& w : result.nodes) {
    move(w.decay);move(w.weight);move(w.bias);move(w.read);
    for (auto& [_,t] : w.extra) move(t);
  }
  for (auto& w : result.regions) for (auto& [_,t] : w.extra) move(t);
  for (auto* group : {&result.input_scale,&result.agg_scale,&result.edge_scale,&result.output_scale})
    for (auto& t : *group) move(t);
  return result;
}
}  // namespace tide
