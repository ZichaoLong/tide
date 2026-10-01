#include "retained_tape.h"
#include <ATen/core/grad_mode.h>
#include <map>
#include <set>
#include <limits>
#include <stdexcept>

namespace tide::device_online {
namespace {
std::vector<at::Tensor*> tensors(ReverseTape& t) {
  std::vector<at::Tensor*> out{&t.state.metadata,&t.state.values,&t.state.count,&t.state.config,&t.state.decay,&t.state.retention,&t.state.clock_policy,
    &t.full.metadata,&t.full.values,&t.full.count,&t.full.kinds,&t.full.weights,&t.full.biases,
    &t.full.extra.lh_kinds,&t.full.extra.lh_weights,&t.full.extra.lh_biases,
    &t.full.extra.swiglu_kinds,&t.full.extra.swiglu_mapping,&t.full.extra.gate,&t.full.extra.up,&t.full.extra.down,
    &t.aggregate.kinds,&t.aggregate.lengths,&t.aggregate.weights,
    &t.control.read,&t.control.raw_full,
    &t.full_values,&t.fiber_meta,&t.fiber_values,&t.fiber_count,&t.sources,&t.source_scales,&t.delivery_scales,
    &t.pending.coordinates,&t.pending.values,&t.pending.valid,&t.pending_count,
    &t.outputs.coordinates,&t.outputs.values,&t.outputs.valid,&t.output_count};
  for(auto& a:t.attention)for(auto* x:{&a.mapping,&a.windows,&a.config,&a.qkv,&a.projection,
      &a.metadata,&a.values,&a.count,&a.key,&a.value,&a.lengths})out.push_back(x);
  for(auto& f:t.fiber) {
    auto& a=f.cache;
    for(auto* x:{&a.mapping,&a.windows,&a.config,&a.qkv,&a.projection,&a.metadata,&a.values,&a.count,&a.key,&a.value,&a.lengths,
        &f.bias,&f.qkv_bias,&f.projection_bias,&f.decay,&f.pool_kinds,&f.pool_lengths,&f.pool_weights})out.push_back(x);
  }
  return out;
}
}
int64_t reverse_tape_bytes(const ReverseTape& source) {
  auto t=source;std::set<const void*> seen;long double bytes=0;
  for(auto* x:tensors(t))if(x->defined()&&seen.insert(x->unsafeGetTensorImpl()).second)
    bytes+=static_cast<long double>(x->numel())*x->element_size();
  if(bytes>std::numeric_limits<int64_t>::max())throw std::invalid_argument("retained tape extent overflow");
  return static_cast<int64_t>(bytes);
}

RetainedTape retain_reverse_tape(const ReverseTape& source,int64_t budget) {
  if(at::GradMode::is_enabled()||!source.graph||!source.fiber_values.defined()||budget<1
      ||source.fiber_values.device().type()!=c10::DeviceType::PrivateUse1)
    throw std::invalid_argument("retained tape requires bounded no-grad NPU forward records");
  RetainedTape out;out.graph=std::make_shared<const Graph>(*source.graph);out.tape=source;out.tape.graph=out.graph.get();
  auto& t=out.tape;
  auto buffers=tensors(t);
  std::map<const void*,at::Tensor> copies;long double bytes=0;
  for(auto* x:buffers)if(x->defined()) {
    if(x->device()!=source.fiber_values.device()||x->requires_grad())throw std::invalid_argument("invalid retained tape ownership");
    const auto key=x->unsafeGetTensorImpl();
    if(copies.emplace(key,at::Tensor()).second)bytes+=static_cast<long double>(x->numel())*x->element_size();
  }
  if(bytes>budget)throw std::invalid_argument("retained tape tensor budget exceeded");
  out.tensor_bytes=static_cast<int64_t>(bytes);
  for(auto* x:buffers)if(x->defined()) {
    auto& copy=copies.at(x->unsafeGetTensorImpl());if(!copy.defined())copy=x->clone();*x=copy;
  }
  return out;
}
} // namespace tide::device_online
