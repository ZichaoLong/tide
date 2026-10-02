#include "full_reverse_pack.h"
#include "cann_api.h"
#include "aclrtlaunch_tide_full_reverse_pack.h"
namespace tide::device_online {
namespace {uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}}
FullReverseBatch append_full_reverse_pack(CannProgram& p,const FullTape& stage,const at::Tensor& gradient,
    const at::Tensor& connected,const at::Tensor& mapping,int64_t local_nodes,
    const at::Tensor& work,const at::Tensor& error,
    const ReverseGatherInput& values,const ReverseGatherInput& gradients) {
  const auto rows=stage.metadata.size(0),nodes=mapping.numel();auto source=at::empty({rows},mapping.options());
  auto tape=stage;tape.metadata=at::empty_like(stage.metadata);tape.count=at::empty_like(stage.count);tape.values=at::empty_like(stage.values);
  FullReverseBatch out{tape,at::empty_like(gradient),at::empty_like(connected),at::empty_like(source),at::empty_like(error)};
  p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_full_reverse_pack)(1,stream,
    ptr(stage.metadata),ptr(stage.count),ptr(connected),ptr(mapping),ptr(out.tape.metadata),ptr(out.tape.count),
    ptr(out.connected),ptr(source),ptr(out.destinations),ptr(out.branch),ptr(work),ptr(error),rows,nodes,local_nodes),
    "pack connected Full reverse owner rows");},
    {stage.metadata,stage.count,connected,mapping,out.tape.metadata,out.tape.count,out.connected,source,out.destinations,out.branch,work,error});
  values.select(p,source,out.tape.values);gradients.select(p,source,out.gradient);
  return out;
}
} // namespace tide::device_online
