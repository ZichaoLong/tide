#include "weighted_flow_internal.h"
namespace tide::device_online {
void append_weighted_region_views(DeviceProgram& p,TaskServices& services,WeightedFlowRegion& r,std::vector<WeightedFlowView>& nodes,
  const at::Tensor& error,const WeightedFlowBudget& remaining){
  const auto samples=nodes.at(r.nodes.at(0)).present.numel(),members=static_cast<int64_t>(r.nodes.size());
  const auto opts=nodes.at(r.nodes.at(0)).content.options().dtype(at::kFloat);
  const auto e=r.config.regional()?*r.config.region_control_width:1,q=r.config.region_score=="att"?*r.config.selection_width:1,width=3+e+2*q;
  auto desc=at::zeros({samples,members,width},opts),time=at::zeros({samples,members},opts.dtype(at::kLong)),present=at::zeros({samples,members},opts.dtype(at::kBool));
  for(int64_t j=0;j<members;++j){auto& n=nodes[r.nodes[j]];const auto& f=n.describe;
    auto target=desc.select(1,j);weighted_copy(p,target.narrow(1,0,1),f.at("w"));weighted_copy(p,target.narrow(1,1,1),f.at("mu"));weighted_copy(p,target.narrow(1,2,1),f.at("score"));
    if(f.count("evidence"))weighted_copy(p,target.narrow(1,3,e),f.at("evidence"));
    if(f.count("x")){weighted_copy(p,target.narrow(1,3+e,q),f.at("x"));weighted_copy(p,target.narrow(1,3+e+q,q),f.at("key"));}
    weighted_copy(p,time.select(1,j),n.times);weighted_copy(p,present.select(1,j),n.present);
  }
  r.input=append_weighted_region_input(p,desc,present,time,r.nodes,r.config,error);
  auto& selection=services.begin();
  r.selection=append_weighted_region(selection,r.input.fields,r.input.ids,r.input.times,r.input.lengths,bind_weighted_region(r.banks,r.config,samples),r.config,error,remaining(selection));
  services.end(selection);
  auto control=at::zeros({samples*members+1},opts),active=at::zeros({samples*members+1},opts.dtype(at::kBool));
  weighted_copy(p,control.narrow(0,0,samples*members),r.selection->controls.reshape({-1}));weighted_copy(p,active.narrow(0,0,samples*members),r.selection->active.reshape({-1}));
  auto controls=at::zeros({samples,members},opts),actives=at::zeros({samples,members},opts.dtype(at::kBool));
  p.index_select(control,0,r.input.inverse.reshape({-1}),controls.reshape({-1}));p.index_select(active,0,r.input.inverse.reshape({-1}),actives.reshape({-1}));
  for(int64_t j=0;j<members;++j){auto& n=nodes[r.nodes[j]];n.control=weighted_columns(p,controls,j,1).reshape({samples});n.active=weighted_columns(p,actives,j,1).reshape({samples});}
}
WeightedFlowView weighted_flow_view(const WeightedFlowNode& n){
  return {n.input.present,n.input.times,n.aggregate->output,n.old,n.proposal,n.describe->fields,n.control,n.active,n.full,n.payload};
}
void append_weighted_region_local(DeviceProgram& p,TaskServices& services,WeightedFlowRegion& r,const WeightedFlowNodes& nodes,
  const at::Tensor& error,const WeightedFlowBudget& remaining){
  std::vector<WeightedFlowView> views(nodes.size());for(auto v:r.nodes)views[v]=weighted_flow_view(*nodes[v]);
  append_weighted_region_views(p,services,r,views,error,remaining);
  for(auto v:r.nodes){nodes[v]->control=views[v].control;nodes[v]->active=views[v].active;}
}
}
