#include "state_owner_tape.h"
#include <ATen/core/grad_mode.h>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

namespace tide::device_online {
namespace {
std::vector<at::Tensor*> tensors(StateOwnerTape& t) {
  std::vector<at::Tensor*> out{&t.state.config,&t.state.decay,&t.state.retention,&t.state.clock_policy,
    &t.read,&t.read_modes,&t.read_kinds,&t.sources};
  auto cache=[&](EventAttentionTape& a) {
    for(auto* x:{&a.mapping,&a.windows,&a.config,&a.qkv,&a.projection,&a.metadata,&a.values,&a.count,&a.key,&a.value,&a.lengths})out.push_back(x);
  };
  for(auto& a:t.attention)cache(a);
  for(auto& f:t.fiber) {
    cache(f.cache);
    for(auto* x:{&f.bias,&f.qkv_bias,&f.projection_bias,&f.decay,&f.pool_kinds,&f.pool_lengths,&f.pool_weights})out.push_back(x);
  }
  return out;
}
void structure(const StateOwnerTape& t) {
  if(t.global_nodes.size()!=size_t(t.layout.nodes)||t.state.metadata.defined()||t.state.values.defined()||t.state.count.defined())
    throw std::invalid_argument("state owner fragment must have compact metadata and no coordinator journals");
  int64_t previous=-1;for(const auto n:t.global_nodes){if(n<=previous)throw std::invalid_argument("invalid reverse owner node map");previous=n;}
  if(!t.state.decay.defined()||t.state.decay.device().type()!=c10::DeviceType::PrivateUse1)
    throw std::invalid_argument("state owner tape requires NPU parameter storage");
}
}
int64_t state_owner_tape_bytes(const StateOwnerTape& source) {
  structure(source);auto t=source;std::set<const void*> seen;long double bytes=0;
  for(auto* x:tensors(t))if(x->defined()&&seen.insert(x->unsafeGetTensorImpl()).second)bytes+=static_cast<long double>(x->nbytes());
  if(bytes>std::numeric_limits<int64_t>::max())throw std::invalid_argument("state owner tape byte overflow");
  return int64_t(bytes);
}
RetainedStateOwnerTape retain_state_owner_tape(const StateOwnerTape& source,int64_t budget) {
  if(at::GradMode::is_enabled()||budget<1)throw std::invalid_argument("state owner retention requires bounded no-grad context");
  const auto bytes=state_owner_tape_bytes(source);
  if(bytes>budget)throw std::invalid_argument("retained state owner tape exceeds tensor budget");
  RetainedStateOwnerTape out{source,bytes};auto buffers=tensors(out.tape);std::map<const void*,at::Tensor> copies;
  for(auto* x:buffers)if(x->defined()) {
    if(x->device()!=source.state.decay.device()||x->requires_grad())throw std::invalid_argument("state owner tape crosses devices or autograd ownership");
    copies.emplace(x->unsafeGetTensorImpl(),at::Tensor());
  }
  for(auto* x:buffers)if(x->defined()) {
    auto& clone=copies.at(x->unsafeGetTensorImpl());if(!clone.defined())clone=x->clone();*x=clone;
  }
  return out;
}
} // namespace tide::device_online
