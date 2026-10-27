#include "weighted_flow_internal.h"
#include "device_launch_tide_weighted_node_pack.h"
#include "device_launch_tide_weighted_region_pack.h"
#include "device_launch_tide_weighted_owner_adopt.h"
namespace tide::device_online {
namespace {uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}}
at::Tensor weighted_columns(DeviceProgram& p,const at::Tensor& x,int64_t start,int64_t count){
  auto out=at::zeros({x.size(0),count},x.options());
  p.index_select(x,1,at::arange(start,start+count,x.options().dtype(at::kLong)),out);return out;
}
WeightedNodeInput append_weighted_node_input(DeviceProgram& p,const ReadyBatch& r,const at::Tensor& classes,
    int64_t node,int64_t samples,int64_t inputs,const at::Tensor& error){
  const auto cap=r.atoms.valid.numel(),width=r.atoms.values.size(1);auto longs=r.fibers.options();
  WeightedNodeInput x;x.payload=at::zeros({samples,cap,width},r.atoms.values.options());x.owners=at::full({samples,cap},-1,longs);
  x.indices=at::full({samples,cap},cap,longs);x.lengths=at::zeros({samples},longs);x.times=at::zeros_like(x.lengths);
  x.present=at::zeros({samples},longs.dtype(at::kBool));x.bank=at::full({samples},-1,longs);
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_node_pack)(1,stream,
    ptr(r.fibers),ptr(r.fiber_offsets),ptr(r.atoms.coordinates),ptr(r.counts),ptr(classes),ptr(x.indices),ptr(x.owners),
    ptr(x.lengths),ptr(x.times),ptr(x.present),ptr(x.bank),ptr(error),node,samples,cap,inputs,classes.size(0)),"weighted complete node fiber plan");},
    {r.fibers,r.fiber_offsets,r.atoms.coordinates,r.counts,classes,x.indices,x.owners,x.lengths,x.times,x.present,x.bank,error});
  auto joined=at::zeros({cap+1,width},r.atoms.values.options());weighted_copy(p,joined.narrow(0,0,cap),r.atoms.values);
  p.index_select(joined,0,x.indices.reshape({-1}),x.payload.reshape({samples*cap,width}));return x;
}
WeightedRegionInput append_weighted_region_input(DeviceProgram& p,const at::Tensor& desc,const at::Tensor& present,
    const at::Tensor& time,const std::vector<int64_t>& nodes,const weighted::RegionalConfig& c,const at::Tensor& error){
  const auto samples=present.size(0),members=present.size(1),width=desc.size(2),rows=samples*members;
  const auto evidence=c.regional()?*c.region_control_width:1,query=c.region_score=="att"?*c.selection_width:1;
  if(width!=3+evidence+2*query)throw std::logic_error("weighted descriptor packing width mismatch");
  auto longs=time.options();auto ids=at::tensor(nodes,at::kLong).to(time.device()),order=at::zeros({rows},longs);
  WeightedRegionInput x;x.ids=at::zeros({samples,members},longs);x.inverse=at::zeros_like(x.ids);x.order=order;
  x.times=at::zeros({samples},longs);x.lengths=at::zeros_like(x.times);
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_region_pack)(1,stream,
    ptr(present),ptr(time),ptr(ids),ptr(order),ptr(x.ids),ptr(x.times),ptr(x.lengths),ptr(x.inverse),ptr(error),samples,members),
    "weighted canonical complete region frames");},{present,time,ids,order,x.ids,x.times,x.lengths,x.inverse,error});
  auto joined=at::zeros({rows+1,width},desc.options()),packed=at::zeros({rows,width},desc.options());
  weighted_copy(p,joined.narrow(0,0,rows),desc.reshape({rows,width}));p.index_select(joined,0,order,packed);
  x.fields={weighted_columns(p,packed,0,1).reshape({samples,members}),weighted_columns(p,packed,1,1).reshape({samples,members}),
    weighted_columns(p,packed,2,1).reshape({samples,members}),weighted_columns(p,packed,3,evidence).reshape({samples,members,evidence}),
    weighted_columns(p,packed,3+evidence,query).reshape({samples,members,query}),weighted_columns(p,packed,3+evidence+query,query).reshape({samples,members,query})};
  return x;
}
void append_weighted_owner_adopt(DeviceProgram& p,const at::Tensor& target,const at::Tensor& present,
    const at::Tensor& times,const at::Tensor& error,bool history){
  const int64_t samples=present.numel(),kind=history;
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_owner_adopt)(1,stream,
    ptr(target),ptr(present),ptr(times),ptr(error),samples,kind),"weighted private owner commit");},{target,present,times,error});
}
void WeightedWindowJournal::save(DeviceProgram& p,const at::Tensor& stage,const std::string& name,const at::Tensor& x){
  if(fields_.count(name)||!x.defined()||!x.is_contiguous()||x.numel()<1)throw std::invalid_argument("invalid weighted window record field");
  const auto cost=static_cast<long double>(stages_)*x.numel()*x.element_size();
  if(cost>remaining_)throw std::invalid_argument("weighted window journal capacity exceeds tensor budget");remaining_-=static_cast<int64_t>(cost);allocated_+=static_cast<int64_t>(cost);
  auto shape=x.sizes().vec();shape.insert(shape.begin(),stages_);auto stored=at::zeros(shape,x.options());
  fields_[name]=stored;p.index_copy(stored,0,stage,x.unsqueeze(0));
}
std::map<std::string,at::Tensor> WeightedWindowJournal::decode(int64_t stages)const{
  if(stages<0||stages>stages_)throw std::logic_error("invalid weighted window journal length");
  std::map<std::string,at::Tensor> out;for(const auto& [name,x]:fields_)out[name]=x.narrow(0,0,stages).to(at::kCPU).clone();return out;
}
}
