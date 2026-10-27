#include "weighted_recurrent_internal.h"
#include "device_launch_tide_weighted_recurrent_prepare.h"
#include <cmath>
namespace tide::device_online {
using namespace recurrent_detail;
WeightedRecurrentSnapshot append_weighted_recurrent_snapshot(DeviceProgram& p,const weighted::RecurrentConfig& c,
    const WeightedRawState& raw,const at::Tensor& times,const at::Tensor& mass,const at::Tensor& owners,const at::Tensor& present,
    int64_t banks,bool positive,const at::Tensor& error,int64_t budget){
  c.validate();if(!raw.integer.defined()||raw.integer.dim()!=2||raw.integer.size(1)<4||banks<1)
    throw std::invalid_argument("invalid recurrent snapshot metadata/banks");
  WeightedRawLayout l(c,raw.integer.size(1)-4);validate_weighted_raw_buffers(l,raw);const auto device=raw.floating.device();const auto rows=raw.floating.size(0);
  if(device.type()!=resident_device_type||raw.floating.scalar_type()!=at::kFloat)throw std::invalid_argument("recurrent snapshot requires explicit accelerator FP32");
  validate_kernel_device(device);check(times,device,at::kLong,{rows});check(mass,device,at::kFloat,{rows});
  check(owners,device,at::kLong,{rows});check(present,device,at::kBool,{rows});check(error,device,at::kInt,{1});
  if(budget<1||rows*(12.L*l.floating_width()+8.L*l.integer_width()+9)>budget)throw std::invalid_argument("recurrent snapshot tensor budget exceeded");
  std::vector<float> rates;if(c.basic())rates={float(c.lambda_short),0};else if(c.matrix_rates.empty())rates.resize(c.heads,0);else for(auto rate:c.matrix_rates){
    if(rate>0&&float(rate)==0)throw std::invalid_argument("positive recurrent passive rate underflows FP32");rates.push_back(float(rate));}
  rates.push_back(float(c.lambda_short));rates.push_back(0);
  for(auto rate:rates)if(!std::isfinite(rate)||rate<0)throw std::invalid_argument("unrepresentable recurrent passive rate");
  if(float(c.lambda_short)<=0)throw std::invalid_argument("unrepresentable recurrent short passive rate");
  auto rate=at::tensor(rates,at::kFloat).to(device),clean=at::zeros_like(raw.floating);
  WeightedRecurrentSnapshot t{l,{at::zeros_like(raw.floating),at::zeros_like(raw.integer)},at::zeros_like(owners),at::zeros_like(raw.floating),at::zeros_like(present)};
  p.copy(t.state.integer,raw.integer);p.copy(t.present,present);
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_recurrent_prepare)(1,stream,
    ptr(raw.floating),ptr(t.state.integer),ptr(times),ptr(mass),ptr(owners),ptr(t.present),ptr(rate),ptr(clean),ptr(t.decay),ptr(t.owners),ptr(error),
    rows,l.floating_width(),l.capacity(),c.width+1,c.matrices(),c.value_width*c.key_width,c.summary_width,c.conv_width(),banks,int64_t(positive),l.horizon()),
    "recurrent exact time snapshot and numeric attenuation");},{raw.floating,t.state.integer,times,mass,owners,t.present,rate,clean,t.decay,t.owners,error});
  p.multiply(clean,t.decay,t.state.floating);return t;
}
}
