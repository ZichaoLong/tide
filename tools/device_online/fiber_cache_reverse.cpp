#include "fiber_reverse.h"
#include "event_reverse.h"
#include "cann_api.h"
#include "aclrtlaunch_tide_cache_bias_merge.h"
#include <stdexcept>

namespace tide::device_online {
namespace {
uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}
std::pair<at::Tensor,at::Tensor> bias_pair(const FiberAttentionTape& t,const CacheCotangents& c) {
  if(c.bias.defined()!=c.bias_connected.defined())throw std::invalid_argument("incomplete cache log-bias cotangent pair");
  if(!c.bias.defined())return {at::empty({1},t.bias.options().dtype(at::kFloat)),at::zeros({t.bias.size(0)},t.bias.options().dtype(at::kBool))};
  if(c.bias.device()!=t.bias.device()||c.bias.scalar_type()!=at::kFloat||c.bias.sizes()!=t.bias.sizes()
      ||c.bias.requires_grad()||!c.bias.is_contiguous()||c.bias_connected.device()!=t.bias.device()
      ||c.bias_connected.scalar_type()!=at::kBool||c.bias_connected.sizes()!=at::IntArrayRef({t.bias.size(0)})
      ||!c.bias_connected.is_contiguous()||c.bias_connected.requires_grad())throw std::invalid_argument("invalid cache log-bias cotangent layout");
  return {c.bias,c.bias_connected};
}
}
CacheCotangents append_fiber_cache_seed(CannProgram& p,const FiberAttentionTape& t,const CacheCotangents& roots,
    const CacheGradient* later,const at::Tensor& error,int64_t budget) {
  const auto& a=t.cache;const int64_t owners=a.samples*a.nodes.size();
  if(budget<2||t.bias.device()!=a.key.device()||t.bias.scalar_type()!=a.key.scalar_type()||!t.bias.is_contiguous()
      ||t.bias.requires_grad()||t.bias.sizes()!=at::IntArrayRef({owners,a.capacity})||4.L*t.bias.numel()+8.L*owners+1024>budget/2.L)
    throw std::invalid_argument("fiber cache boundary budget/layout rejected");
  auto out=later?append_cache_bridge(p,a,roots,*later,error,budget/2):append_cache_seed(p,a,roots,error,budget/2);
  auto left=bias_pair(t,roots),right=later?bias_pair(t,*later):left;
  out.bias=at::empty(t.bias.sizes(),t.bias.options().dtype(at::kFloat));out.bias_connected=at::empty_like(left.second);
  p.zero(out.bias);p.zero(out.bias_connected);
  for(int64_t phase:{0,1})p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_cache_bias_merge)(phase?32:1,stream,
    ptr(a.lengths),ptr(left.first),ptr(left.second),ptr(right.first),ptr(right.second),ptr(out.bias),ptr(out.bias_connected),ptr(error),
    owners,a.capacity,int64_t(later!=nullptr),phase),"merge same-fiber log-bias boundary adjoints");},
    {a.lengths,left.first,left.second,right.first,right.second,out.bias,out.bias_connected,error});
  return out;
}
} // namespace tide::device_online
