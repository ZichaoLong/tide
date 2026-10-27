#include "weighted_recurrent_internal.h"
#include "device_launch_tide_weighted_recurrent_reverse.h"
#include "device_launch_tide_weighted_row_accumulate.h"
namespace tide::device_online {
using namespace recurrent_detail;
WeightedRecurrentGradient append_weighted_recurrent_vjp(DeviceProgram& p,const WeightedRecurrentTape& t,
    const std::map<std::string,WeightedRowAdjoint>& roots,const at::Tensor& error,int64_t budget){
  const auto initial_bytes=p.retained_tensor_bytes();
  const auto& l=t.snapshot.layout;const auto rows=t.snapshot.owners.numel(),count=int64_t(l.fields().size());const auto device=t.snapshot.owners.device();
  check(error,device,at::kInt,{1});auto options=t.snapshot.state.floating.options();
  if(budget<1||rows*(28.L*l.floating_width()+7.L*count)>budget)throw std::invalid_argument("recurrent VJP tensor budget exceeded");
  for(const auto& [name,root]:roots){if(!t.fields.count(name))throw std::invalid_argument("inapplicable recurrent output root: "+name);
    check(root.value,device,at::kFloat,t.fields.at(name).sizes());check(root.connected,device,at::kBool,{rows});}
  std::vector<WeightedRowRoot> numerical_roots;
  auto raw_shape=t.snapshot.state.floating.sizes();auto bypass=at::zeros(raw_shape,options),bypass_roots=at::zeros({rows,count},options.dtype(at::kBool));
  std::optional<WeightedRawUpdateGradient> updated;
  if(t.update){auto packed=at::zeros(raw_shape,options),flags=at::zeros_like(bypass_roots);p.zero(packed);p.zero(flags);
    for(int64_t j=0;j<count;++j){const auto& field=l.fields()[j];auto found=roots.find(field.name);if(found==roots.end())continue;
      insert(p,packed,field.offset,found->second.value);insert(p,flags,j,found->second.connected.reshape({rows,1}));}
    updated=append_weighted_raw_update_vjp(p,*t.update,packed,flags,error,budget);bypass=updated->old_values;bypass_roots=updated->old_connected;
    for(const auto& [name,index]:t.outputs){const auto& field=l.field(name);int64_t j=0;while(l.fields()[j].name!=name)++j;
      numerical_roots.push_back({index,columns(p,updated->proposal,field.offset,t.graph->value(index).size(1)),columns(p,updated->proposal_connected,j,1).reshape({rows})});}
  }else for(const auto& [name,root]:roots)numerical_roots.push_back({t.outputs.at(name),root.value,root.connected});
  auto g=t.graph->reverse(p,numerical_roots,error,budget);
  auto packed=at::zeros(raw_shape,options),flags=at::zeros_like(bypass_roots);p.zero(packed);p.zero(flags);std::vector<int64_t> fields;
  for(int64_t j=0;j<count;++j){const auto& f=l.fields()[j];fields.insert(fields.end(),{f.offset,f.width,int64_t(f.ragged)});
    if(f.name=="value")continue;const auto& root=g.rows.at(t.inputs.at(f.name));insert(p,packed,f.offset,root.value);insert(p,flags,j,root.connected.reshape({rows,1}));}
  auto schema=at::tensor(fields,at::kLong).to(device);
  WeightedRecurrentGradient result{at::zeros(raw_shape,options),at::zeros_like(flags),g.rows.at(t.inputs.at("mass")),g.rows.at(t.inputs.at("content")),std::move(g.parameters)};
  const auto snapshot=t.snapshot;
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_recurrent_reverse)(1,stream,
    ptr(snapshot.owners),ptr(snapshot.state.integer),ptr(snapshot.decay),ptr(schema),ptr(packed),ptr(flags),ptr(bypass),ptr(bypass_roots),
    ptr(result.raw),ptr(result.raw_connected),ptr(error),rows,l.floating_width(),l.capacity(),count),"recurrent effective and raw-identity field adjoints");},
    {snapshot.owners,snapshot.state.integer,snapshot.decay,schema,packed,flags,bypass,bypass_roots,result.raw,result.raw_connected,error});
  if(updated){auto value=updated->mass.reshape({rows,1}),connected=updated->mass_connected,owner=snapshot.owners,bar=result.mass.value,flag=result.mass.connected;
    p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_row_accumulate)(1,stream,
      ptr(owner),ptr(value),ptr(connected),ptr(bar),ptr(flag),ptr(error),rows,1),"recurrent numerical and raw receive adjoints");},{owner,value,connected,bar,flag,error});}
  long double unused_bytes=0;
  for(const auto& [name,value]:t.parameters)if(!result.parameters.count(name)){
    unused_bytes+=4.L*value.numel()+value.size(0);result.parameters[name]={at::zeros_like(value),at::zeros({value.size(0)},options.dtype(at::kBool))};}
  if(p.retained_tensor_bytes()-initial_bytes+unused_bytes>budget)throw std::invalid_argument("complete recurrent VJP tensor budget exceeded");
  return result;
}
}
