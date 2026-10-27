#pragma once
#include "weighted_region.h"
#include "weighted_region_layout.h"
#include "device_launch_tide_weighted_allocation_status.h"
#include <ATen/core/grad_mode.h>
#include <cmath>
namespace tide::device_online::region_detail {
inline uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}
inline void check(const at::Tensor& x,at::IntArrayRef shape,at::ScalarType type,at::Device device){
  if(!x.defined()||x.sizes()!=shape||x.scalar_type()!=type||x.device()!=device||!x.is_contiguous()||x.requires_grad())
    throw std::invalid_argument("invalid resident weighted region buffer");
}
inline void admit(long double required,int64_t budget){
  if(budget<1||required>budget)throw std::invalid_argument("resident weighted region scratch budget exceeded");
}
inline void status(DeviceProgram& p,const at::Tensor& rows,const at::Tensor& error){
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_allocation_status)(
    1,stream,ptr(rows),ptr(error),rows.numel()),"region all-row decision");},{rows,error});
}
inline int64_t mode(const weighted::RegionalConfig& c){
  const std::vector<std::string> names{"regional_total","regional_total_zero","regional_threshold","regional_threshold_hard",
    "constrained","hard_selection","direct_attention","unit_attention","pooled_attention","proportional"};
  for(size_t i=0;i<names.size();++i)if(c.selection_mode==names[i])return i;
  throw std::invalid_argument("unknown weighted resident region mode");
}
struct Shape {
  int64_t rows,width,e,q,h,kind,k;
  bool context,attention,received,allocated;
  float fixed,tau,alpha_local,alpha_query,inverse_sqrt;
  Shape(const at::Tensor& x,const weighted::RegionalConfig& c){
    c.validate();
    if(!x.defined()||x.dim()!=2||x.size(0)<1||x.size(1)<1||at::GradMode::is_enabled()
       ||x.device().type()!=resident_device_type)throw std::invalid_argument("resident region requires accelerator rows and explicit no-grad");
    validate_kernel_device(x.device());check(x,x.sizes(),at::kFloat,x.device());
    rows=x.size(0);width=x.size(1);c.capacity(width);kind=mode(c);k=c.k.value_or(0);
    context=c.region_control_mode=="context";attention=c.region_score=="att";received=c.query_weight=="received";
    e=c.regional()?*c.region_control_width:1;q=attention?*c.selection_width:1;h=c.hidden_width;
    allocated=kind==weighted_region::Total||kind==weighted_region::Threshold||kind==weighted_region::Constrained;
    fixed=c.total()?c.total_fraction.value_or(0):c.eta.value_or(0);tau=c.allocation_temperature.value_or(0);
    alpha_local=c.alpha_loc.value_or(1);alpha_query=c.alpha_sel.value_or(0);inverse_sqrt=1/std::sqrt(float(q));
    for(float v:{fixed,tau,alpha_local,alpha_query,inverse_sqrt})if(!std::isfinite(v))
      throw std::invalid_argument("resident region configuration exceeds FP32 domain");
    if((allocated||c.attention())&&tau<=0)throw std::invalid_argument("resident temperature underflowed in FP32");
  }
};
inline long double head_bytes(int64_t rows,int64_t in,int64_t hidden,int64_t out){
  return rows*(4.L*(in+2.L*hidden*in+2.L*out*hidden+3.L*hidden+2.L*out)+20);
}
inline void head_shape(const WeightedHeadParameters& w,int64_t rows,int64_t in,int64_t hidden,int64_t out,
                       const at::Tensor& owners,at::Device device){
  if(!w.matrix.defined()||w.matrix.dim()!=3||w.matrix.size(0)<1)throw std::invalid_argument("missing resident region parameter bank");
  const auto banks=w.matrix.size(0);check(w.matrix,{banks,hidden,in},at::kFloat,device);
  check(w.hidden_bias,{banks,hidden},at::kFloat,device);check(w.projection,{banks,out,hidden},at::kFloat,device);
  check(w.bias,{banks,out},at::kFloat,device);check(owners,{rows},at::kLong,device);
}
inline WeightedRegionFields zeros(const Shape& s,const at::TensorOptions& options){
  return {at::zeros({s.rows,s.width},options),at::zeros({s.rows,s.width},options),at::zeros({s.rows,s.width},options),
    at::zeros({s.rows,s.width,s.e},options),at::zeros({s.rows,s.width,s.q},options),at::zeros({s.rows,s.width,s.q},options)};
}
}
