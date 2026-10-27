#include "weighted_flow_vjp.h"
namespace tide::device_online {
WeightedRegionAdjoint append_weighted_region_views_vjp(DeviceProgram& p,TaskServices& services,const WeightedFlowRegion& r,
    const std::vector<WeightedFlowView>& nodes,const std::map<int64_t,WeightedRowAdjoint>& controls,
    const at::Tensor& error,const WeightedFlowBudget& remaining){
  using namespace weighted_reverse;
  if(!r.selection)throw std::invalid_argument("weighted region VJP needs actual selection tape");
  const auto rows=r.input.times.numel(),members=int64_t(r.nodes.size());const auto options=r.selection->controls.options();
  auto original=at::zeros({rows,members},options),packed=at::zeros_like(original),joined=at::zeros({rows*members+1},options);
  p.zero(original);p.zero(joined);auto roots=zeros(p,at::zeros({rows,1},options));
  for(int64_t j=0;j<members;++j){auto found=controls.find(r.nodes[j]);if(found==controls.end())continue;
    auto local=zeros(p,at::zeros({rows,1},options));const auto& node=nodes.at(r.nodes[j]);
    add(p,local,found->second,node.bank,error);weighted_copy(p,original.narrow(1,j,1),local.value);
    // OR connectivity independently of numerical zero (dummy values are zero).
    add(p,roots,{at::zeros({rows,1},options),local.connected},node.bank,error);
  }
  weighted_copy(p,joined.narrow(0,0,rows*members),original.reshape({-1}));p.index_select(joined,0,r.input.order,packed.reshape({-1}));
  auto& worker=services.begin();auto g=append_weighted_region_vjp(worker,*r.selection,packed,roots.connected,error,remaining(worker));services.end(worker);
  WeightedRegionAdjoint result;
  auto head=[&](const std::string& name,const WeightedHeadGradient& part){
    parameter(p,result.parameters,r.banks,name+".matrix",part.parameters.matrix,part.owner_connected);
    parameter(p,result.parameters,r.banks,name+".hidden_bias",part.parameters.hidden_bias,part.owner_connected);
    parameter(p,result.parameters,r.banks,name+".projection",part.parameters.projection,part.owner_connected);
    parameter(p,result.parameters,r.banks,name+".bias",part.parameters.bias,part.owner_connected);};
  if(g.context_head)head("region.context",*g.context_head);if(g.query_head)head("region.query",*g.query_head);
  const std::string names[]{"w","mu","score","evidence","x","key"};
  const at::Tensor values[]{g.fields.received,g.fields.caps,g.fields.score,g.fields.evidence,g.fields.current,g.fields.key};
  for(int64_t field=0;field<6;++field){const auto width=values[field].numel()/(rows*members);
    auto saved=at::zeros({rows*members+1,width},options),ordered=at::zeros({rows*members,width},options);
    weighted_copy(p,saved.narrow(0,0,rows*members),values[field].reshape({rows*members,width}));
    p.index_select(saved,0,r.input.inverse.reshape({-1}),ordered);
    auto flag=weighted_columns(p,g.connected,field,1).reshape({rows});
    for(int64_t j=0;j<members;++j){const auto id=r.nodes[j];const auto& node=nodes.at(id);if(!node.describe.count(names[field]))continue;
      auto local=zeros(p,node.describe.at(names[field]));auto value=at::zeros_like(local.value);
      weighted_copy(p,value,ordered.reshape({rows,members,width}).select(1,j));
      add(p,local,{value,flag},node.bank,error);result.describe[id][names[field]]=local;
    }
  }
  remaining(p);return result;
}
WeightedRegionAdjoint append_weighted_flow_region_vjp(DeviceProgram& p,TaskServices& services,const WeightedFlowRegion& r,
    const WeightedFlowNodes& nodes,const std::map<int64_t,WeightedRowAdjoint>& controls,
    const at::Tensor& error,const WeightedFlowBudget& remaining){
  std::vector<WeightedFlowView> views(nodes.size());for(auto v:r.nodes)views[v]=weighted_flow_view(*nodes[v]);
  return append_weighted_region_views_vjp(p,services,r,views,controls,error,remaining);
}
}
