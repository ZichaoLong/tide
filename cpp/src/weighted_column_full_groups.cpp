// KV §4: selected queries only, private banks and unweighted G message contents.
#include "weighted_column_groups.h"
#include "tide/packing_work.h"
namespace tide::weighted::columns {
using detail::require;using detail::weight;
namespace {
std::vector<Tensor> evaluate(const std::vector<KVFullRequest>& requests,const std::vector<size_t>& ids,const std::vector<Message>& selected){
  const auto& c=requests[ids[0]].config;const std::string prefix="full_program.content.";
  std::vector<QueryInputs> qs;std::vector<Tensor> mixed;std::vector<const NodeWeights*> owners;
  for(auto i:ids){const auto& r=requests[i];
    qs.push_back(query_inputs(r.window,r.window.bank_layout,r.window.bank,selected[i].rows,false,c.window));
    mixed.push_back(at::zeros_like(qs.back().u));owners.push_back(r.weights);}
  auto chosen=groups::active(qs);
  if(!chosen.empty()){
    std::vector<Tensor> inputs,matrices;for(auto i:chosen){inputs.push_back(normalize(qs[i].u,c.epsilon));matrices.push_back(weight(*owners[i],prefix+"query"));}
    auto projected=project_nodes(inputs,matrices);std::vector<std::vector<Tensor>> operands;std::vector<PositionQuery> positions;
    bool positional=false;for(const auto& f:c.full_frequencies)positional|=!f.empty();for(auto s:c.distance_slopes)positional|=s!=0;
    for(size_t n=0;n<chosen.size();++n){auto i=chosen[n];const auto& query=qs[i];const auto b=query.u.size(0),length=query.u.size(1),dh=c.width/c.heads;
      auto q=projected[n].reshape({b,length,c.heads,dh}).transpose(1,2);
      auto k=query.raw.at("keys").reshape({b,-1,c.heads,dh}).transpose(1,2),v=query.raw.at("values").reshape({b,-1,c.heads,dh}).transpose(1,2);
      if(positional){const auto& r=requests[ids[i]];auto frequencies=c.full_frequencies;if(frequencies.empty())frequencies.resize(c.heads);
        positions.push_back({q,k,query.age,r.window.layout.select(selected[ids[i]].rows),r.window.bank_layout,
                             frequencies,c.position_scale,c.distance_slopes});
        operands.push_back({v,query.weights});
      }else operands.push_back({q,k,v,query.weights});
    }
    if(positional){auto logits=position_logits_many(positions);
      for(size_t n=0;n<operands.size();++n)operands[n].insert(operands[n].begin(),logits[n]);}
    auto values=contract_nodes(positional?Contraction::logits:Contraction::attention,operands);
    for(size_t n=0;n<chosen.size();++n)mixed[chosen[n]]=values[n].transpose(1,2).reshape(qs[chosen[n]].u.sizes());
  }
  auto project=[&](const std::vector<Tensor>& xs,const std::string& name){std::vector<Tensor> matrices;
    for(auto owner:owners)matrices.push_back(weight(*owner,prefix+name));return project_nodes(xs,matrices);};
  std::vector<Tensor> inputs;for(size_t i=0;i<qs.size();++i)inputs.push_back(mixed[i]*qs[i].weights.sum(-1).clamp_max(1).unsqueeze(-1));
  auto out=project(inputs,"out");std::vector<Tensor> xs,ns;
  for(size_t i=0;i<qs.size();++i){xs.push_back(qs[i].u+c.alpha_att*out[i]);ns.push_back(normalize(xs.back(),c.epsilon));}
  auto gate=project(ns,"gate"),up=project(ns,"up");inputs.clear();
  for(size_t i=0;i<qs.size();++i)inputs.push_back(at::silu(gate[i])*up[i]);
  auto down=project(inputs,"down");
  for(size_t i=0;i<qs.size();++i)out[i]=(xs[i]+c.alpha_ff*down[i]).flatten(0,1).index_select(0,qs[i].reverse);
  return out;
}
}
std::vector<Message> full_many(const std::vector<KVFullRequest>& requests,Index max_workspace_bytes){
  std::vector<Message> out(requests.size());std::vector<size_t> ids;std::vector<Layout> queries(requests.size());
  for(size_t i=0;i<requests.size();++i){const auto& r=requests[i];const auto& w=r.window;const auto& control=r.controls;
    require(r.weights&&supported(r.config),"unsupported KV Full group or parameter owner");
    require(control.sizes()==w.mass.sizes()&&control.device()==w.mass.device()&&control.scalar_type()==w.mass.scalar_type()
      &&at::isfinite(control).all().item<bool>()&&!(control<0).any().item<bool>(),"invalid KV column controls");
    auto index=(control.detach()>0).nonzero().flatten();out[i]={host_ids(index),control.index_select(0,index),at::empty({0,r.config.width},w.content.options())};
    if(!out[i].rows.empty()){ids.push_back(i);queries[i]=w.layout.select(out[i].rows);}}
  auto same=[&](size_t a,size_t b){const auto& x=requests[a];const auto& y=requests[b];
    return same_configuration(x.config,y.config)&&groups::same_payload(x.window.content,y.window.content)
      &&groups::shape(queries[a])==groups::shape(queries[b])&&groups::shape(x.window.bank_layout)==groups::shape(y.window.bank_layout);};
  auto cost=[&](size_t i){const auto& r=requests[i];return groups::bytes(*r.weights,r.config,queries[i],r.window.bank_layout,r.window.content.element_size());};
  for(const auto& batch:groups::plan(ids,max_workspace_bytes,same,cost)){
    work::packing::bucket("kv_full_nodes",groups::shape(queries[batch[0]]).second,batch.size());
    if(batch.size()==1){auto i=batch[0];const auto& r=requests[i];out[i]=full(*r.weights,r.config,r.window,r.controls);continue;}
    auto values=evaluate(requests,batch,out);for(size_t k=0;k<batch.size();++k)out[batch[k]].content=std::move(values[k]);}
  return out;
}
}
