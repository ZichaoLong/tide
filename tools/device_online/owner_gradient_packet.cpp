#include "device_backend.h"
#include "owner_gradient_packet.h"
#include "parameter_vjp.h"
#include "device_launch_tide_owner_gradient_pack.h"
#include "device_launch_tide_owner_gradient_reduce.h"
#include <cstring>
#include <stdexcept>
#include <ATen/core/grad_mode.h>
namespace tide::device_online {
namespace {
uint8_t* ptr(const at::Tensor& t){return static_cast<uint8_t*>(t.data_ptr());}
int64_t address(const at::Tensor& t) {const auto a=reinterpret_cast<uint64_t>(t.data_ptr());int64_t out;std::memcpy(&out,&a,sizeof(out));return out;}
at::Tensor table(const std::vector<int64_t>& values,at::Device d) {
  return at::tensor(values.empty()?std::vector<int64_t>{0}:values,at::kLong).to(d);
}
void buffer(const at::Tensor& x,at::Device d,at::ScalarType dtype,int64_t size) {
  if(!x.defined()||x.device()!=d||x.scalar_type()!=dtype||x.dim()!=1||x.numel()!=size||!x.is_contiguous()||x.requires_grad())
    throw std::invalid_argument("invalid owner gradient buffer");
}
}
void append_owner_gradient_pack(DeviceProgram& p,const std::vector<ParameterContribution>& input,
    const at::Tensor& values,const at::Tensor& connected,const at::Tensor& error,int64_t budget) {
  std::vector<int64_t> descriptors,tiles{0};std::vector<at::Tensor> keep{values,connected,error};int64_t used=0;
  if(at::GradMode::is_enabled()||!values.defined()||values.device().type()!=tide::device_online::resident_device_type)
    throw std::invalid_argument("owner gather requires explicit no-grad NPU buffers");
  buffer(error,values.device(),at::kInt,1);
  if(budget<1||48.L*(input.size()+1)>budget)throw std::invalid_argument("owner gather metadata budget exceeded");
  for(const auto& c:input) {
    if(!c.values.defined()||!c.connected.defined()||c.values.device()!=values.device()||c.connected.device()!=values.device()||c.values.scalar_type()!=at::kFloat
        ||c.connected.scalar_type()!=at::kBool||c.connected.numel()!=1||!c.values.is_contiguous()||!c.connected.is_contiguous()
        ||c.values.requires_grad()||c.connected.requires_grad())
      throw std::invalid_argument("invalid owner gather source");
    const auto n=c.values.numel();if(n<1||n>values.numel()-used)throw std::invalid_argument("owner gather extent exceeds buffer");
    descriptors.insert(descriptors.end(),{address(c.values),address(c.connected),used,n});tiles.push_back(tiles.back()+(n+255)/256);used+=n;
    keep.push_back(c.values);keep.push_back(c.connected);
  }
  buffer(values,values.device(),at::kFloat,std::max<int64_t>(1,used));
  buffer(connected,values.device(),at::kBool,std::max<int64_t>(1,input.size()));
  const auto desc=table(descriptors,values.device()),offsets=table(tiles,values.device());
  keep.push_back(desc);keep.push_back(offsets);p.zero(values);p.zero(connected);
  const int64_t count=input.size(),tasks=tiles.back();
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_owner_gradient_pack)(32,stream,
    ptr(desc),ptr(offsets),ptr(values),ptr(connected),ptr(error),count,tasks),"pack canonical owner contributions");},keep);
}
void append_owner_gradient_reduce(DeviceProgram& p,const std::vector<int64_t>& owners,const std::vector<int64_t>& refs,
    const std::vector<int64_t>& tiles,const at::Tensor& partials,const at::Tensor& partial_on,const ParameterVjp& output,
    const at::Tensor& error,int64_t budget) {
  if(at::GradMode::is_enabled()||!output.values.defined()||output.values.device().type()!=tide::device_online::resident_device_type
      ||budget<1||8.L*(owners.size()+refs.size()+tiles.size()+3)>budget||owners.size()!=output.owners.size()*4
      ||output.offsets.size()!=output.owners.size()||refs.size()%2||tiles.size()!=output.owners.size()+1||tiles.front()!=0)
    throw std::invalid_argument("invalid owner reduction metadata/budget");
  const auto d=output.values.device();buffer(error,d,at::kInt,1);buffer(partials,d,at::kFloat,partials.numel());
  buffer(partial_on,d,at::kBool,partial_on.numel());buffer(output.values,d,at::kFloat,output.values.numel());
  buffer(output.connected,d,at::kBool,std::max<int64_t>(1,output.owners.size()));
  for(size_t i=0;i<output.owners.size();++i) {
    const auto first=owners[i*4],last=owners[i*4+1],offset=owners[i*4+2],size=owners[i*4+3];
    if(first<0||last<first||last>int64_t(refs.size()/2)||size!=output.owners[i].value.numel()||size<1
        ||offset!=output.offsets[i]||offset < -1||offset<0&&last>first
        ||offset>=0&&(size>output.values.numel()||offset>output.values.numel()-size)
        ||tiles[i+1]!=tiles[i]+(offset>=0?(size+255)/256:0))throw std::invalid_argument("invalid owner reduction extent");
    for(int64_t j=first;j<last;++j)if(refs[j*2]<0||size>partials.numel()||refs[j*2]>partials.numel()-size
        ||refs[j*2+1]<0||refs[j*2+1]>=partial_on.numel())throw std::invalid_argument("invalid owner reduction reference");
  }
  auto desc=table(owners,output.values.device()),references=table(refs,output.values.device()),offsets=table(tiles,output.values.device());
  p.zero(output.values);p.zero(output.connected);const int64_t count=output.owners.size(),tasks=tiles.back();
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_owner_gradient_reduce)(32,stream,
    ptr(desc),ptr(references),ptr(offsets),ptr(partials),ptr(partial_on),ptr(output.values),ptr(output.connected),ptr(error),count,tasks),
    "reduce canonical parameter owners in declared order");},
    {desc,references,offsets,partials,partial_on,output.values,output.connected,error});
}
} // namespace tide::device_online
