#include "weighted_flow_internal.h"
#include "device_launch_tide_weighted_payload_guard.h"
namespace tide::device_online {
at::Tensor append_weighted_payload(DeviceProgram& p,const at::Tensor& control,const at::Tensor& full,
    const at::Tensor& active,at::ScalarType dtype,const at::Tensor& error){
  const auto rows=full.size(0),width=full.size(1)+1;
  if(dtype!=at::kFloat&&dtype!=at::kHalf)throw std::invalid_argument("weighted payload storage must be FP32/FP16");
  auto values=at::zeros({rows,width},full.options());
  weighted_copy(p,values.narrow(1,0,1),control.reshape({rows,1}));weighted_copy(p,values.narrow(1,1,width-1),full);
  if(dtype==at::kFloat)return values;
  auto result=at::empty(values.sizes(),values.options().dtype(dtype)),rounded=at::empty_like(values);
  p.cast(values,result);p.cast(result,rounded);
  p.kernel([=](void* stream){auto ptr=[](const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());};
    check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_payload_guard)(1,stream,ptr(values),ptr(rounded),ptr(active),ptr(error),rows,width),
      "weighted half payload finite and positive-support guard");},{values,rounded,active,error});
  return result;
}
}
