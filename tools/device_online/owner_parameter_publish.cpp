#include "sharded_parameter_banks.h"
#include "cann_api.h"
#include "aclrtlaunch_tide_owner_parameter_publish.h"
#include <ATen/core/grad_mode.h>
#include <cstring>
#include <stdexcept>
namespace tide::device_online {
namespace {uint8_t* ptr(const at::Tensor& t){return static_cast<uint8_t*>(t.data_ptr());}}
void append_owner_parameter_publish(CannProgram& p,const std::vector<ParameterWrite>& writes,
    const at::Tensor& values,const at::Tensor& error,int64_t budget) {
  if(at::GradMode::is_enabled()||!values.defined()||values.dim()!=1||!values.is_contiguous()
      ||values.device().type()!=c10::DeviceType::PrivateUse1||values.scalar_type()!=at::kFloat||values.requires_grad()
      ||!error.defined()||error.device()!=values.device()||error.scalar_type()!=at::kInt||error.sizes()!=at::IntArrayRef({1})
      ||!error.is_contiguous()||error.requires_grad()||budget<1||64.L*(writes.size()+1)>budget)
    throw std::invalid_argument("invalid sharded publication buffers/budget");
  std::vector<int64_t> descriptors,tiles{0};std::vector<at::Tensor> keep{values,error};
  for(const auto& write:writes) {
    const auto& x=write.destination.values;const auto payload=write.destination.payload_dtype;
    if(!x.defined()||x.device()!=values.device()||x.requires_grad()||x.numel()<1
        ||(x.scalar_type()!=at::kFloat&&x.scalar_type()!=at::kHalf)||(payload!=at::kFloat&&payload!=at::kHalf)
        ||(payload==at::kFloat&&x.scalar_type()==at::kHalf)||write.offset<0||x.numel()>values.numel()-write.offset
        ||(!x.is_contiguous()&&(x.dim()!=2||x.stride(1)!=1||x.stride(0)<x.size(1))))
      throw std::invalid_argument("invalid sharded publication destination/offset");
    const int64_t rows=x.is_contiguous()?1:x.size(0),cols=x.numel()/rows,stride=x.is_contiguous()?cols:x.stride(0);
    const auto pointer=reinterpret_cast<uint64_t>(x.data_ptr());int64_t address;std::memcpy(&address,&pointer,sizeof(address));
    descriptors.insert(descriptors.end(),{write.offset,address,rows,cols,stride,x.scalar_type()==at::kHalf,payload==at::kHalf});
    tiles.push_back(tiles.back()+rows*((cols+255)/256));keep.push_back(x);
  }
  auto table=at::tensor(descriptors.empty()?std::vector<int64_t>(7,0):descriptors,at::kLong).to(values.device());
  auto offsets=at::tensor(tiles,at::kLong).to(values.device());keep.push_back(table);keep.push_back(offsets);
  const int64_t count=writes.size(),tasks=tiles.back();
  p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_owner_parameter_publish)(32,stream,
    ptr(table),ptr(offsets),ptr(values),ptr(error),count,tasks),"publish canonical owners to local parameter aliases");},keep);
}
} // namespace tide::device_online
