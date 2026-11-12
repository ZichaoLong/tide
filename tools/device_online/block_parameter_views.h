#pragma once
#include <ATen/ATen.h>
#include <string>
namespace tide::device_online {
// Only source-function banks may use explicit zero padding. Other numerical
// roles permit shape-only views, preserving their canonical owner's elements.
inline at::Tensor block_parameter_view(const std::string& role,const at::Tensor& bank,at::IntArrayRef original){
  int64_t elements=1;for(auto x:original)elements*=x;
  if(elements==bank.numel())return bank.reshape(original);
  if(role.rfind("source.",0)!=0||bank.dim()!=int64_t(original.size())||bank.dim()<1||bank.dim()>2)
    throw std::invalid_argument("block parameter shape cannot be represented by this numerical role");
  auto out=bank;for(int64_t axis=0;axis<bank.dim();++axis){if(original[axis]<1||original[axis]>bank.size(axis))
      throw std::invalid_argument("block source parameter exceeds padded bank extent");out=out.narrow(axis,0,original[axis]);}
  return out;
}
inline at::Tensor block_parameter_pack(const std::string& role,const at::Tensor& original,at::IntArrayRef shape){
  auto out=at::zeros(shape,original.options());block_parameter_view(role,out,original.sizes()).copy_(original);return out;
}
}
