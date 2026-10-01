#include "training_internal.h"
#include <ATen/core/grad_mode.h>
#include <set>
#include <stdexcept>

namespace tide::training_detail {
void no_grad() {
  if(at::GradMode::is_enabled())throw std::invalid_argument("explicit resident VJP requires no-grad; compute consumer cotangents separately");
}
Model freeze_model(Model model,at::Device device,std::vector<Version>& versions) {
  std::map<const void*,Tensor> copies;std::set<const void*> storage;
  for(const auto& owner:model.parameters(false).owners()) {
    const auto& x=owner.value;
    if((x.scalar_type()!=at::kFloat&&x.scalar_type()!=at::kHalf)||(!x.device().is_cpu()&&x.device()!=device))
      throw std::invalid_argument("resident training parameters require CPU/session NPU FP32/FP16");
    if(!storage.insert(x.storage().unsafeGetStorageImpl()).second)
      throw std::invalid_argument("resident training refuses distinct owners sharing storage");
    versions.push_back({x,x._version(),x.const_data_ptr()});
    auto copy=x.detach().cpu().clone();
    if(!at::isfinite(copy).all().item<bool>())throw std::invalid_argument("nonfinite initial resident parameter");
    copy.set_requires_grad(x.requires_grad());copies.emplace(x.unsafeGetTensorImpl(),copy);
  }
  auto copy=[&](Tensor& x){x=copies.at(x.unsafeGetTensorImpl());};
  for(auto& w:model.nodes){copy(w.decay);copy(w.weight);copy(w.bias);copy(w.read);for(auto& [_,x]:w.extra)copy(x);}
  for(auto& w:model.regions)for(auto& [_,x]:w.extra)copy(x);
  for(auto* group:{&model.input_scale,&model.agg_scale,&model.edge_scale,&model.output_scale})for(auto& x:*group)copy(x);
  return model;
}
Continuation freeze_continuation(Continuation q) {
  auto copy=[](Tensor& x){x=x.detach().cpu().clone();};
  for(auto& [_,s]:q.states){copy(s.value);for(auto& [__,x]:s.slots)copy(x);}
  for(auto& [_,h]:q.history)for(auto& [__,x]:h.tensors)copy(x);
  for(auto& a:q.pending)copy(a.value);return q;
}
void restore_parameters(Model& model,const ResidentTrainingCheckpoint& c) {
  if(c.schema!=1||c.generation<0||c.next_token<0)throw std::invalid_argument("invalid resident training checkpoint schema/progress");
  const auto all=model.parameters(false),trainable=model.parameters(true);
  if(all.alias_partitions()!=c.aliases||trainable.names()!=c.trainable||all.owners().size()!=c.parameters.size())
    throw std::invalid_argument("resident training checkpoint alias/trainable ownership mismatch");
  for(const auto& owner:all.owners()) {
    auto it=c.parameters.find(owner.canonical);
    if(it==c.parameters.end())throw std::invalid_argument("missing checkpoint parameter");
    const auto& x=it->second;
    if(!x.defined()||!x.device().is_cpu()||x.scalar_type()!=owner.value.scalar_type()||x.sizes()!=owner.value.sizes()
        ||x.requires_grad()||!at::isfinite(x).all().item<bool>())throw std::invalid_argument("invalid checkpoint parameter");
  }
  for(const auto& owner:all.owners())owner.value.copy_(c.parameters.at(owner.canonical));
}
} // namespace tide::training_detail
