#include "full_shard_pack.h"
#include "cann_api.h"
#include "aclrtlaunch_tide_full_shard_pack.h"
namespace tide::device_online {
namespace {uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}}
FullShardBatch append_full_shard_pack(CannProgram& p,const ActionBatch& actions,const at::Tensor& content,
    const at::Tensor& comparison,const at::Tensor& mapping,int64_t local_nodes,const at::Tensor& work,const at::Tensor& error) {
  const auto rows=actions.values.size(0),width=actions.values.size(1),nodes=mapping.numel();
  auto source=at::empty({rows},mapping.options());
  FullShardBatch out{{at::empty_like(actions.coordinates),at::empty_like(actions.values),at::empty_like(actions.valid)},
    at::empty_like(content),at::empty_like(comparison),at::empty_like(source),at::empty_like(error)};
  p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_full_shard_pack)(1,stream,
    ptr(actions.coordinates),ptr(actions.valid),ptr(mapping),ptr(out.actions.coordinates),ptr(out.actions.valid),
    ptr(source),ptr(out.destinations),ptr(out.branch),ptr(work),ptr(error),rows,nodes,local_nodes),"pack selected Full shard actions");},
    {actions.coordinates,actions.valid,mapping,out.actions.coordinates,out.actions.valid,source,out.destinations,out.branch,work,error});
  for(const auto& pair:{std::make_pair(actions.values,out.actions.values),std::make_pair(content,out.content),std::make_pair(comparison,out.comparison)}) {
    auto padded=at::zeros({rows+1,width},pair.first.options());
    p.copy(padded.narrow(0,0,rows),pair.first);p.index_select(padded,0,source,pair.second);
  }
  return out;
}
} // namespace tide::device_online
