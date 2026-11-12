#include "block_read_selection_internal.h"
#include "weighted_saved_tensors.h"
#include "device_launch_tide_block_selection_roots.h"
#include "device_launch_tide_block_selection_flags.h"
#include "device_launch_tide_weighted_row_accumulate.h"

namespace tide::device_online {
using namespace block_selection_detail;
namespace {
std::vector<at::Tensor> region_saved(const WeightedRegionTape& t){
  std::vector<at::Tensor> out{t.fields.received,t.fields.caps,t.fields.score,t.fields.evidence,t.fields.current,t.fields.key,
    t.lengths,t.ids,t.times,t.context_input,t.query_input,t.statistics,t.context_rows,t.query_rows,t.context_values,
    t.query_values,t.scores,t.bounds,t.parameters,t.controls,t.active,t.history_times,t.history_present};
  for(const auto* h:{&t.context_head,&t.query_head})if(*h){const auto& x=**h;
    out.insert(out.end(),{x.input,x.owners,x.indices,x.preactivation,x.hidden,x.output,
      x.rows.matrix,x.rows.hidden_bias,x.rows.projection,x.rows.bias});}
  if(t.allocation){const auto& x=*t.allocation;out.insert(out.end(),{x.controls,x.slopes,x.statistics,x.branches,x.lengths,x.row_errors});}
  if(t.sending){const auto& x=*t.sending;out.insert(out.end(),{x.controls,x.details,x.statistics,x.branches,x.lengths});}
  return out;
}
}
BlockReadSelectionGradient append_block_read_selection_vjp(DeviceProgram& p,const BlockReadSelectionTape& t,
    const at::Tensor& gradient,const at::Tensor& roots,const at::Tensor& error,int64_t budget,const BlockReadObservations& observations){
  if(!t.describe.graph||t.frames<1||t.nodes<1||t.regions<1||t.members<1)
    throw std::invalid_argument("missing block read-selection tape");
  const auto rows=t.frames*t.nodes,rr=t.frames*t.regions,members=t.members;
  check(gradient,error,at::kFloat,{t.frames,t.nodes});check(roots,error,at::kBool,{t.frames,t.nodes});
  auto borrowed=saved_tensors(t.query);auto describe_saved=t.describe.graph->saved_tensors();
  borrowed.insert(borrowed.end(),describe_saved.begin(),describe_saved.end());
  auto regional_saved=region_saved(t.region);borrowed.insert(borrowed.end(),regional_saved.begin(),regional_saved.end());
  for(const auto& x:{t.owners,t.present,t.node_times,t.order,t.inverse,t.ids,t.times,t.lengths,t.node_region_row,
    t.controls,t.active,gradient,roots,error,t.region.fields.received,t.region.fields.caps,t.region.fields.score,
    t.region.fields.evidence,t.region.fields.current,t.region.fields.key})borrowed.push_back(x);
  for(const auto* x:{&observations.memory,&observations.total}){borrowed.push_back(x->value);borrowed.push_back(x->connected);}
  for(const auto& [name,x]:observations.describe){borrowed.push_back(x.value);borrowed.push_back(x.connected);}
  const WeightedStorageIncrement storage(p,std::move(borrowed));
  auto remaining=[&](){admit(storage.bytes(),budget);return budget-storage.bytes();};
  admit(4.L*rr*members+rr+rows*6,budget);
  auto packed=at::zeros({rr,members},gradient.options()),flags=at::zeros({rr},roots.options());
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_block_selection_roots)(1,stream,
    ptr(t.order),ptr(gradient),ptr(roots),ptr(packed),ptr(flags),ptr(error),rr,members,rows),"block independent control roots");},
    {t.order,gradient,roots,packed,flags,error});
  auto regional=append_weighted_region_vjp(p,t.region,packed,flags,error,remaining());
  auto node_flags=at::zeros({rows,6},roots.options());
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_block_selection_flags)(1,stream,
    ptr(t.present),ptr(t.node_region_row),ptr(regional.connected),ptr(node_flags),ptr(error),0,rows,rr),"block region root support unpack");},
    {t.present,t.node_region_row,regional.connected,node_flags,error});
  std::map<std::string,WeightedRowAdjoint> describe_roots;
  const std::vector<std::string> names{"w","mu","score","evidence","x","key"};
  const std::vector<at::Tensor> fields{regional.fields.received,regional.fields.caps,regional.fields.score,
    regional.fields.evidence,regional.fields.current,regional.fields.key};
  for(int64_t j=0;j<6;++j)if(t.describe.fields.count(names[j])){
    auto bar=gather(p,fields[j].reshape({rr*members,-1}),t.inverse);
    describe_roots[names[j]]={bar,columns(p,node_flags,j,1).reshape({rows})};
  }
  auto merge=[&](WeightedRowAdjoint left,const WeightedRowAdjoint& right,int64_t width){
    if(right.value.defined()!=right.connected.defined())throw std::invalid_argument("observation requires value and independent flags");
    if(!right.value.defined())return left;
    check(right.value,error,at::kFloat,{rows,width});check(right.connected,error,at::kBool,{rows});
    auto flags=at::zeros_like(right.connected);p.logical_and(right.connected,t.present,flags);
    const auto owners=t.owners;
    p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_row_accumulate)(1,stream,
      ptr(owners),ptr(right.value),ptr(flags),ptr(left.value),ptr(left.connected),ptr(error),rows,width),
      "independent Read observation root");},{owners,right.value,flags,left.value,left.connected,error});
    return left;
  };
  for(const auto& [name,root]:observations.describe){
    if(!t.describe.fields.count(name))throw std::invalid_argument("inapplicable Describe observation: "+name);
    const auto width=t.describe.fields.at(name).size(1);
    if(!describe_roots.count(name)){describe_roots[name]={at::zeros({rows,width},gradient.options()),at::zeros({rows},roots.options())};
      p.zero(describe_roots[name].value);p.zero(describe_roots[name].connected);}
    describe_roots[name]=merge(describe_roots.at(name),root,width);
  }
  auto described=append_weighted_describe_vjp(p,t.describe,describe_roots,error,remaining());
  const auto ds=t.query.memory.size(2);auto memory=at::zeros({rows,2*ds},gradient.options()),total=at::zeros({rows,2},gradient.options());
  auto query_roots=at::zeros({rows,4},roots.options());
  for(int64_t channel=0;channel<2;++channel){
    auto mem=described.inputs.at(channel?"long_memory":"short_memory"),mass=described.inputs.at(channel?"long_total":"short_total");
    auto channel_root=[&](const WeightedRowAdjoint& root,int64_t width,bool vector){
      if(root.value.defined()!=root.connected.defined())throw std::invalid_argument("Mem root requires value and channel flags");
      if(!root.value.defined())return WeightedRowAdjoint{};
      if(vector)check(root.value,error,at::kFloat,{rows,2,width});else check(root.value,error,at::kFloat,{rows,2});
      check(root.connected,error,at::kBool,{rows,2});
      return WeightedRowAdjoint{columns(p,root.value.reshape({rows,2*width}),channel*width,width),columns(p,root.connected,channel,1).reshape({rows})};
    };
    mem=merge(mem,channel_root(observations.memory,ds,true),ds);
    mass=merge(mass,channel_root(observations.total,1,false),1);
    put(p,memory,channel*ds,mem.value);put(p,total,channel,mass.value);
    put(p,query_roots,channel,mem.connected.reshape({rows,1}));put(p,query_roots,2+channel,mass.connected.reshape({rows,1}));
  }
  auto query=append_weighted_kv_query_vjp(p,t.query,memory.reshape({rows,2,ds}),total,query_roots,error,remaining());
  auto content=at::zeros_like(query.input),connected=at::zeros_like(query.input_connected);
  p.copy(content,query.input);p.add(content,described.inputs.at("content").value);
  const auto dflags=described.inputs.at("content").connected;
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_block_selection_flags)(1,stream,
    ptr(query.input_connected),ptr(t.node_region_row),ptr(dflags),ptr(connected),ptr(error),1,rows,rr),"block Read/Describe shared input root merge");},
    {query.input_connected,t.node_region_row,dflags,connected,error});
  admit(storage.bytes(),budget);
  const auto mass=described.inputs.at("mass");
  return {query.raw,query.raw_connected,mass.value.reshape({t.frames,t.nodes}),mass.connected.reshape({t.frames,t.nodes}),
    content.reshape({t.frames,t.nodes,-1}),connected.reshape({t.frames,t.nodes}),query.parameters,
    query.query_connected,query.key_connected,query.value_connected,std::move(described.parameters),regional.context_head,regional.query_head};
}
}
