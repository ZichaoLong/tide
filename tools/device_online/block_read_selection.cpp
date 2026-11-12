#include "block_read_selection_internal.h"
#include "device_launch_tide_block_selection_plan.h"
#include "device_launch_tide_block_node_gate.h"
#include <limits>
#include <set>

namespace tide::device_online {
using namespace block_selection_detail;
SettleBlockPlan lower_settle_blocks(const SettleGraph& spec,int64_t first_position,int64_t rows,int64_t bytes){
  if(spec.binding()!="ports")throw std::invalid_argument("resident Settle blocks require explicit ports binding");
  std::vector<int64_t> owners;for(const auto& n:spec.graph().nodes)owners.push_back(n.region);
  std::vector<std::array<int64_t,3>> edges;for(const auto& e:spec.graph().edges)edges.push_back({e.source,e.target,e.delay});
  return make_settle_block_plan(owners,spec.ranks(),edges,spec.graph().inputs,first_position,spec.stride(),rows,bytes);
}
BlockReadSelectionTape append_block_read_selection(DeviceProgram& p,const SettleBlockPlan& layout,
    const weighted::KVConfig& kv,const weighted::RegionalConfig& config,const BlockReadSelectionInput& x,
    const BlockReadSelectionParameters& weights,const at::Tensor& error,int64_t budget){
  kv.validate();config.validate();
  if(!error.defined()||error.dim()!=1||error.numel()!=1||error.scalar_type()!=at::kInt)
    throw std::invalid_argument("block read-selection requires an int32 error flag");
  if(!x.present.defined()||x.present.dim()!=2||x.present.size(0)<1||
     layout.member_offsets.size()<2||layout.members.empty())throw std::invalid_argument("block read-selection requires nonempty KV sum/qry layout");
  if(kv.memory_read=="sum"&&(weights.query.query.defined()||weights.query.key.defined()||weights.query.value.defined()))
    throw std::invalid_argument("KV sum Read does not bind query parameters");
  const int64_t frames=x.present.size(0),nodes=layout.members.size(),regions=layout.member_offsets.size()-1;
  if(frames>std::numeric_limits<int64_t>::max()/nodes||frames>std::numeric_limits<int64_t>::max()/regions)
    throw std::invalid_argument("block read-selection row count overflow");
  const int64_t rows=frames*nodes,rr=frames*regions;
  int64_t members=0;
  if(layout.member_offsets[0]!=0||layout.member_offsets.back()!=nodes||std::set<int64_t>(layout.members.begin(),layout.members.end()).size()!=size_t(nodes))
    throw std::invalid_argument("invalid block region membership");
  for(auto v:layout.members)if(v<0||v>=nodes)throw std::invalid_argument("invalid block region node");
  for(int64_t r=0;r<regions;++r){const auto n=layout.member_offsets[r+1]-layout.member_offsets[r];
    if(n<1)throw std::invalid_argument("empty block region");members=std::max(members,n);}
  check(x.present,error,at::kBool,{frames,nodes});check(x.mass,error,at::kFloat,{frames,nodes});
  check(x.content,error,at::kFloat,{frames,nodes,kv.width});check(x.times,error,at::kLong,{frames,nodes});
  check(x.node_owners,error,at::kLong,{frames,nodes});check(error,error,at::kInt,{1});
  if(rr>std::numeric_limits<int64_t>::max()/members)throw std::invalid_argument("block region slots overflow");
  if(!x.bank&&(!x.read.floating.defined()||x.read.floating.dim()!=2||
     !x.read.integer.defined()||x.read.integer.dim()!=2||x.read.integer.size(1)<=4||
     x.read.floating.size(0)!=rows||x.read.integer.size(0)!=rows))
    throw std::invalid_argument("block raw snapshot row count differs or region slots overflow");
  if(!x.bank&&(x.read.floating.device()!=error.device()||x.read.integer.device()!=error.device()))
    throw std::invalid_argument("block raw snapshot device differs");
  const int64_t e=config.regional()?*config.region_control_width:1,q=config.region_score=="att"?*config.selection_width:1;
  const WeightedStorageIncrement storage(p,borrowed(x,weights,error));
  admit(64.L*rows+rr*(32.L+members*(24.L+4.L*(3+e+2*q))),budget);
  auto remaining=[&](){auto used=storage.bytes();admit(used,budget);return budget-used;};
  BlockReadSelectionTape t;t.frames=frames;t.nodes=nodes;t.regions=regions;t.members=members;
  const auto longs=x.times.options(),booleans=x.present.options();
  t.owners=at::zeros({rows},longs);t.present=at::zeros({rows},booleans);t.node_times=at::zeros({rows},longs);
  t.order=at::zeros({rr,members},longs);t.inverse=at::zeros({rows},longs);t.ids=at::zeros_like(t.order);
  t.times=at::zeros({rr},longs);t.lengths=at::zeros_like(t.times);t.node_region_row=at::zeros({rows},longs);
  auto offsets=at::tensor(layout.member_offsets,at::kLong).to(error.device());
  auto membership=at::tensor(layout.members,at::kLong).to(error.device());
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_block_selection_plan)(1,stream,
    ptr(x.present),ptr(x.times),ptr(x.node_owners),ptr(offsets),ptr(membership),ptr(t.owners),ptr(t.present),ptr(t.node_times),
    ptr(t.order),ptr(t.inverse),ptr(t.ids),ptr(t.times),ptr(t.lengths),ptr(t.node_region_row),ptr(error),frames,nodes,regions,members),
    "resident complete block candidate plan");},{x.present,x.times,x.node_owners,offsets,membership,t.owners,t.present,t.node_times,t.order,t.inverse,t.ids,t.times,t.lengths,t.node_region_row,error});
  if(x.bank)t.memory=append_block_kv_memory(p,kv,*x.bank,t.node_times,x.content.reshape({rows,kv.width}),t.owners,t.present,weights.query,error,remaining());
  else if(kv.memory_read=="sum")t.memory=append_weighted_kv_sum(p,kv,x.read,t.node_times,t.present,error,remaining());
  else t.memory=append_weighted_kv_query(p,kv,x.read,t.node_times,x.content.reshape({rows,kv.width}),t.owners,t.present,weights.query,error,remaining());
  auto memory=t.memory_values().reshape({rows,2*kv.summary_width});
  WeightedDescribeInput d{x.mass.reshape({rows}),x.content.reshape({rows,kv.width}),columns(p,memory,0,kv.summary_width),
    columns(p,memory,kv.summary_width,kv.summary_width),columns(p,t.memory_totals(),0,1).reshape({rows}),columns(p,t.memory_totals(),1,1).reshape({rows})};
  t.describe=append_weighted_describe(p,d,t.owners,weights.describe,config,kv.epsilon,error,remaining());
  auto pack=[&](const std::string& name,int64_t width){
    if(!t.describe.fields.count(name))return at::zeros({rr,members,width},x.mass.options());
    return gather(p,t.describe.fields.at(name),t.order).reshape({rr,members,width});};
  WeightedRegionFields f{pack("w",1).reshape({rr,members}),pack("mu",1).reshape({rr,members}),pack("score",1).reshape({rr,members}),
    pack("evidence",e),pack("x",q),pack("key",q)};
  auto w=weights.region;auto region_ids=at::arange(regions,longs).repeat({frames});
  auto owners=[&](const at::Tensor& source){check(source,error,at::kLong,{regions});auto target=at::zeros({rr},longs);p.index_select(source,0,region_ids,target);return target;};
  if(config.region_control_mode=="context")w.context_owners=owners(w.context_owners);
  if(config.region_score=="att")w.query_owners=owners(w.query_owners);
  t.region=append_weighted_region(p,f,t.ids,t.times,t.lengths,w,config,error,remaining());
  t.controls=gather(p,t.region.controls.reshape({rr*members,1}),t.inverse).reshape({frames,nodes});
  t.active=gather(p,t.region.active.reshape({rr*members,1}),t.inverse).reshape({frames,nodes});
  admit(storage.bytes(),budget);return t;
}
void append_block_node_gate(DeviceProgram& p,const BlockDispatcher& dispatcher,const at::Tensor& active){
  if(active.dim()!=2)throw std::invalid_argument("node gate needs explicit frame,node support");
  const auto tasks=dispatcher.tasks(),ids=dispatcher.selection(),meta=dispatcher.metadata(),emit=dispatcher.emitted(),error=dispatcher.error();
  check(active,error,at::kBool,active.sizes());const auto slots=tasks.size(0),capacity=ids.numel(),frames=active.size(0),nodes=active.size(1);
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_block_node_gate)(1,stream,
    ptr(tasks),ptr(ids),ptr(meta),ptr(active),ptr(emit),ptr(error),slots,capacity,frames,nodes),"resident per-target Full gate");},
    {tasks,ids,meta,active,emit,error});
}
}
