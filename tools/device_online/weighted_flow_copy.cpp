#include "weighted_flow_internal.h"
#include <numeric>
#include <set>
namespace tide::device_online {
namespace {
// Static layout only. Values remain on device; no numerical capture/prepass.
std::pair<at::Tensor,at::Tensor> indexed_view(const at::Tensor& x){
  const auto count=x.storage().nbytes()/x.element_size();
  auto flat=x.as_strided({static_cast<int64_t>(count)},{1},0);
  std::vector<int64_t> indices(x.numel());
  for(int64_t i=0;i<x.numel();++i){auto j=i,offset=x.storage_offset();
    for(int64_t axis=x.dim();axis-->0;){offset+=(j%x.size(axis))*x.stride(axis);j/=x.size(axis);}
    if(offset<0||uint64_t(offset)>=count)throw std::invalid_argument("weighted static view exceeds storage");indices[i]=offset;
  }
  if(std::set<int64_t>(indices.begin(),indices.end()).size()!=indices.size())
    throw std::invalid_argument("weighted copy requires nonoverlapping static views");
  return {flat,at::tensor(indices,at::kLong).to(x.device())};
}
}
void weighted_copy(DeviceProgram& p,const at::Tensor& target,const at::Tensor& source){
  if(target.sizes()!=source.sizes()||target.scalar_type()!=source.scalar_type()||target.device()!=source.device())
    throw std::invalid_argument("weighted copy shape/dtype/device mismatch");
  if(target.is_contiguous()&&source.is_contiguous()){p.copy(target,source);return;}
  auto packed=source;
  if(!source.is_contiguous()){
    auto [flat,indices]=indexed_view(source);packed=at::empty(source.sizes(),source.options());
    p.index_select(flat,0,indices,packed.reshape({-1}));
  }
  if(target.is_contiguous())p.copy(target,packed);
  else {auto [flat,indices]=indexed_view(target);p.index_copy(flat,0,indices,packed.reshape({-1}));}
}
}
