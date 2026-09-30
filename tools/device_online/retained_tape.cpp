#include "retained_tape.h"
#include <ATen/core/grad_mode.h>
#include <map>
#include <stdexcept>

namespace tide::device_online {
RetainedTape retain_reverse_tape(const ReverseTape& source,int64_t budget) {
  if(at::GradMode::is_enabled()||!source.graph||!source.fiber_values.defined()||budget<1
      ||source.fiber_values.device().type()!=c10::DeviceType::PrivateUse1)
    throw std::invalid_argument("retained tape requires bounded no-grad NPU forward records");
  RetainedTape out;out.graph=std::make_shared<const Graph>(*source.graph);out.tape=source;out.tape.graph=out.graph.get();
  auto& t=out.tape;
  std::vector<at::Tensor*> buffers{&t.state.metadata,&t.state.values,&t.state.count,&t.state.config,&t.state.decay,&t.state.retention,&t.state.clock_policy,
    &t.full.metadata,&t.full.values,&t.full.count,&t.full.kinds,&t.full.weights,&t.full.biases,
    &t.full_values,&t.fiber_meta,&t.fiber_values,&t.fiber_count,&t.sources,&t.source_scales,&t.delivery_scales,
    &t.pending.coordinates,&t.pending.values,&t.pending.valid,&t.pending_count,
    &t.outputs.coordinates,&t.outputs.values,&t.outputs.valid,&t.output_count};
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
