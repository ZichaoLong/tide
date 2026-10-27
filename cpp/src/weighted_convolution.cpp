#include "tide/weighted_convolution.h"
#include "weighted_detail.h"
#include <limits>

namespace tide::weighted {
using namespace detail;
namespace {
Index horizon(Index size,Index stride){
  require(size>0&&stride>0&&(size-1)<=(std::numeric_limits<Index>::max()-1)/stride,"invalid convolution size/stride/capacity");
  return (size-1)*stride;
}
void query(const ConvCache& b,Index time,const Tensor& z,Index limit){
  require(time>=0&&z.dim()==1&&z.is_floating_point()&&at::isfinite(z).all().item<bool>(),"invalid convolution query");
  validate_cache(b,z.numel(),limit);
  require(b.values.device()==z.device()&&b.values.scalar_type()==z.scalar_type(),"convolution query/cache dtype/device mismatch");
}
}
void validate_cache(const ConvCache& b,Index width,Index limit){
  const auto n=b.times.numel();
  require(b.times.scalar_type()==at::kLong&&b.times.dim()==1&&!b.times.requires_grad()
    &&b.values.sizes()==at::IntArrayRef({n,width})&&b.values.is_floating_point()&&b.times.device()==b.values.device()
    &&n<=limit+1,"invalid convolution cache layout");
  require((b.times>=0).all().item<bool>()&&(n<2||(b.times.slice(0,1)>b.times.slice(0,0,n-1)).all().item<bool>())
    &&at::isfinite(b.values).all().item<bool>(),"invalid convolution cache values/times");
}
Tensor convolve(const ConvCache& b,Index time,const Tensor& z,const Tensor& w,const Tensor& bias,Index stride){
  require(w.dim()==2&&w.size(1)==z.numel()&&bias.sizes()==z.sizes()&&w.device()==z.device()
    &&w.scalar_type()==z.scalar_type()&&bias.device()==z.device()&&bias.scalar_type()==z.scalar_type(),"invalid convolution parameters");
  auto limit=horizon(w.size(0),stride);query(b,time,z,limit);
  auto result=bias+w[0]*z;
  if(w.size(0)==1||b.times.numel()==0)return result;
  auto taps=time-stride*at::arange(1,w.size(0),b.times.options());
  auto matches=(taps.unsqueeze(1)==b.times.unsqueeze(0)).to(z.scalar_type());
  return result+(w.slice(0,1)*at::matmul(matches,b.values)).sum(0);
}
ConvCache conv_write(const ConvCache& b,Index time,const Tensor& receive,const Tensor& z,Index size,Index stride){
  const auto limit=horizon(size,stride);query(b,time,z,limit);
  require(receive.dim()==0&&receive.device()==z.device()&&receive.scalar_type()==z.scalar_type()
    &&at::isfinite(receive).item<bool>()&&receive.item<double>()>=0&&receive.item<double>()<=1,"invalid convolution write weight");
  if(receive.item<double>()==0)return b;
  require(!b.times.numel()||time>b.times[-1].item<Index>(),"convolution write time must increase");
  auto mask=(b.times>=std::max<Index>(0,time-limit))&(b.times<time);
  return {at::cat({b.times.index({mask}),at::full({1},time,b.times.options())}),
          at::cat({b.values.index({mask}),(receive*z).unsqueeze(0)})};
}
std::pair<Tensor,Tensor> conv_effective(const ConvCache& b,Index time,Index size,Index stride){
  require(time>=0,"negative convolution projection time");const auto limit=horizon(size,stride);
  require(b.values.dim()==2,"invalid convolution values");validate_cache(b,b.values.size(1),limit);
  const auto start=std::max<Index>(0,time-limit);
  auto slots=at::arange(time-start+1,b.times.options())+start;
  auto matches=(slots.unsqueeze(1)==b.times.unsqueeze(0)).to(b.values.scalar_type());
  return {slots,at::matmul(matches,b.values)};
}
}
