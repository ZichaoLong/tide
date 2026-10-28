#include "tide/parameter_bank.h"
#include <limits>

namespace tide {
namespace {
int64_t elements(const at::Tensor& x,const std::vector<int64_t>& shape,int64_t offset){
  int64_t count=1;
  for(auto n:shape){
    if(n<1||count>std::numeric_limits<int64_t>::max()/n)
      throw std::invalid_argument("invalid parameter bank shape");
    count*=n;
  }
  if(!x.defined()||!x.device().is_cpu()||(x.scalar_type()!=at::kFloat&&x.scalar_type()!=at::kHalf)
      ||offset<0||offset>x.numel()||count>x.numel()-offset)
    throw std::invalid_argument("weighted bank requires explicit CPU FP32/FP16 parameter and exact slice");
  return count;
}
}
at::Tensor copy_parameter_bank(const at::Tensor& x,const std::vector<int64_t>& shape,
                              int64_t offset,at::Device device,int64_t& available){
  const auto count=elements(x,shape,offset);
  if(available<0||count>available/int64_t(sizeof(float)))
    throw std::invalid_argument("weighted parameter bank exceeds tensor budget before allocation");
  // A forced copy also isolates CPU tests/same-device consumers; avoid a second
  // device clone after conversion. A failed copy does not consume the allowance.
  auto value=x.detach().reshape({-1}).narrow(0,offset,count).reshape(shape).to(device,at::kFloat,false,true);
  available-=count*int64_t(sizeof(float));
  return value;
}
at::Tensor ParameterBankPool::bind(const at::Tensor& x,const std::vector<int64_t>& shape,
                                 int64_t offset,at::Device device,int64_t& available){
  const auto count=elements(x,shape,offset);
  if(available<0)throw std::invalid_argument("negative parameter bank pool allowance");
  const auto id=reinterpret_cast<std::uintptr_t>(x.unsafeGetTensorImpl());
  const auto source=sources_.find(id);
  if(source!=sources_.end()&&source->second.version!=x._version())
    throw std::invalid_argument("frozen parameter bank source version changed");
  const auto key=std::make_tuple(id,offset,count,int(device.type()),int(device.index()));
  const auto found=banks_.find(key);
  if(found!=banks_.end())return found->second.reshape(shape);
  auto result=copy_parameter_bank(x,shape,offset,device,available);
  // Retain the source handle so a released TensorImpl address cannot alias a
  // different later parameter. Equal independent tensors remain separate.
  sources_.emplace(id,Source{x,x._version()});banks_.emplace(key,result);return result;
}
}
