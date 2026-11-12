// Pure int64 layout planning and no-grad packing; no per-node numerical graph.
#include "weighted_kv_tensor.h"
#include "tide/packing_work.h"
#include <ATen/core/grad_mode.h>

namespace tide::weighted::columns::tensor {
namespace {
struct Map {Tensor index,restore,times;Index total=0;std::vector<Index> lengths;};
Map layout_map(const std::vector<Request>& requests,Index width,bool query,std::vector<Tensor>* publish){
  Map result;const Index batch=requests[0].query_layout.samples(),nodes=requests.size();
  const auto device=requests[0].window->content.device();
  for(const auto& r:requests){const Index n=query?r.window->layout.times.size():r.bank_layout->times.size();
    result.lengths.push_back(n);result.total+=n;}
  std::vector<Index> index(nodes*batch*width,result.total),restore(result.total,nodes*batch*width),times(nodes*batch*width,0);
  Index base=0;
  for(Index g=0;g<nodes;++g){const auto& r=requests[g];const auto& layout=query?r.query_layout:*r.bank_layout;std::vector<Index> outputs;
    for(Index b=0;b<batch;++b)for(Index j=layout.offsets[b];j<layout.offsets[b+1];++j){
      const Index slot=(g*batch+b)*width+j-layout.offsets[b],source=query?r.rows[j]:j;
      index[slot]=base+source;restore[base+source]=slot;times[slot]=layout.times[j];outputs.push_back(slot);
    }
    if(publish)publish->push_back(indices(outputs,device));base+=result.lengths[g];
  }
  result.index=indices(index,device).reshape({nodes,batch,width});result.restore=indices(restore,device);
  result.times=indices(times,device).reshape({nodes,batch,width});return result;
}
}
Packed pack(const std::vector<Request>& requests,bool differentiable){
  at::NoGradGuard guard;Packed p;p.nodes=requests.size();p.batch=requests[0].query_layout.samples();p.queries=p.keys=1;
  for(const auto& r:requests)for(Index b=0;b<p.batch;++b){
    p.queries=std::max(p.queries,r.query_layout.offsets[b+1]-r.query_layout.offsets[b]);
    p.keys=std::max(p.keys,r.bank_layout->offsets[b+1]-r.bank_layout->offsets[b]);}
  auto qm=layout_map(requests,p.queries,true,&p.publish),km=layout_map(requests,p.keys,false,nullptr);
  if(requests[0].coordinates.reason=="rebased"){
    std::vector<Index> qt(p.nodes*p.batch*p.queries,0),kt(p.nodes*p.batch*p.keys,0);
    for(Index g=0;g<p.nodes;++g){const auto& r=requests[g];
      for(Index b=0;b<p.batch;++b){
        for(Index j=r.query_layout.offsets[b];j<r.query_layout.offsets[b+1];++j)
          qt[(g*p.batch+b)*p.queries+j-r.query_layout.offsets[b]]=r.coordinates.query[j];
        for(Index j=r.bank_layout->offsets[b];j<r.bank_layout->offsets[b+1];++j)
          kt[(g*p.batch+b)*p.keys+j-r.bank_layout->offsets[b]]=r.coordinates.key[j];
      }
    }
    p.query_position=indices(qt,requests[0].window->content.device()).reshape({p.nodes,p.batch,p.queries});
    p.key_position=indices(kt,requests[0].window->content.device()).reshape({p.nodes,p.batch,p.keys});
    work::packing::bucket("kv_tensor_rope_rebased",p.batch*p.queries,p.nodes);
  }
  p.age=qm.times.unsqueeze(-1)-km.times.unsqueeze(-2);
  p.visible=(requests[0].before?p.age>0:p.age>=0)&(qm.index<qm.total).unsqueeze(-1)&(km.index<km.total).unsqueeze(-2);
  if(requests[0].config.window)p.visible=p.visible&(p.age<requests[0].config.window);
  auto data=[&](const std::string& name,const Map& map,auto get){
    Role role;role.name=name;role.lengths=map.lengths;role.restore=map.restore;
    for(const auto& r:requests)role.originals.push_back(get(r));
    auto tensors=role.originals;auto zero=tensors[0].sizes().vec();zero[0]=1;tensors.push_back(at::zeros(zero,tensors[0].options()));
    auto shape=map.index.sizes().vec();shape.insert(shape.end(),zero.begin()+1,zero.end());
    role.value=at::cat(tensors).index_select(0,map.index.flatten()).reshape(shape);
    if(differentiable)role.value.set_requires_grad(true);
    p.values[name]=role.value;p.roles.push_back(std::move(role));
  };
  data("$u",qm,[](const Request& r){return r.window->content;});
  if(!requests[0].full)data("$mass",qm,[](const Request& r){return r.window->mass;});
  data("$weights",km,[](const Request& r){return r.bank->at("weights");});
  if(!requests[0].full&&requests[0].config.memory_read=="sum")data("$summaries",km,[](const Request& r){return r.bank->at("summaries");});
  else if(requests[0].active){
    data("$keys",km,[](const Request& r){return r.bank->at("keys");});
    data("$values",km,[](const Request& r){return r.bank->at("values");});
  }
  for(const auto& name:parameters(requests[0])){Role role;role.name=name;role.parameter=true;
    for(const auto& r:requests)role.originals.push_back(detail::weight(*r.owner,name));
    role.shared=true;for(const auto& value:role.originals)role.shared&=value.is_same(role.originals[0]);
    if(role.shared){role.value=role.originals[0].detach();p.shared.insert(name);}
    else role.value=at::stack(role.originals);
    if(differentiable)role.value.set_requires_grad(true);
    p.values[name]=role.value;p.roles.push_back(std::move(role));
  }
  work::packing::bucket(requests[0].full?"kv_tensor_full":"kv_tensor_read",p.batch*p.queries,p.nodes);
  return p;
}
}
