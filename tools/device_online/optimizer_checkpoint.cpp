#include "device_optimizer.h"
#include <ATen/core/grad_mode.h>
#include <stdexcept>

namespace tide::device_online {
ResidentOptimizerState DeviceOptimizer::snapshot() const {
  if(at::GradMode::is_enabled())throw std::invalid_argument("optimizer checkpoint requires no-grad");
  auto copy=[](const at::Tensor& x){return x.detach().cpu().clone();};
  return {copy(values_),copy(first_),copy(second_),copy(maximum_),copy(steps_),copy(corrections_)};
}
void DeviceOptimizer::restore(const ResidentOptimizerState& s) {
  if(at::GradMode::is_enabled())throw std::invalid_argument("optimizer restore requires no-grad");
  const std::vector<std::pair<at::Tensor,at::Tensor>> fields={{values_,s.values},{first_,s.first},
    {second_,s.second},{maximum_,s.maximum},{steps_,s.steps},{corrections_,s.corrections}};
  for(const auto& [target,input]:fields) {
    if(!input.defined()||!input.device().is_cpu()||input.scalar_type()!=target.scalar_type()
        ||input.sizes()!=target.sizes()||!input.is_contiguous()||input.requires_grad())
      throw std::invalid_argument("optimizer checkpoint tensor layout mismatch");
    if(input.is_floating_point()&&!at::isfinite(input).all().item<bool>())
      throw std::invalid_argument("optimizer checkpoint contains nonfinite values");
  }
  if((s.steps<0).any().item<bool>()||(s.second<0).any().item<bool>()||(s.maximum<0).any().item<bool>()
      ||(s.corrections<0).any().item<bool>()||(s.corrections>1).any().item<bool>())
    throw std::invalid_argument("invalid optimizer checkpoint slots/counters");
  for(int64_t i=0;i<count_;++i) {
    const auto step=s.steps[i].item<int64_t>();
    if((identity_.offsets[i]<0&&step!=0)||(step==0&&s.corrections[i].ne(0).any().item<bool>())
        ||(kind_==DeviceOptimizerKind::sgd&&s.corrections[i].ne(0).any().item<bool>())
        ||(kind_==DeviceOptimizerKind::adamw&&step>0&&s.corrections[i].le(0).any().item<bool>()))
      throw std::invalid_argument("optimizer checkpoint counter/correction mismatch");
  }
  // Numerical parameter values and named aliases are checked by the public
  // training owner before this transaction. Runtime errors invalidate that owner.
  for(const auto& [target,input]:fields)target.copy_(input);
}
} // namespace tide::device_online
