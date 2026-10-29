#include "tide/graph_checkpoint.h"
#include "graph_checkpoint_internal.h"
#include "tide/ops.h"
#include "tide/ownership.h"
#include <ATen/core/grad_mode.h>
namespace tide {
using namespace checkpoint_detail;
namespace {
// This follows ParameterRegistry::add_model's public named ownership inventory.
// Replace every handle by TensorImpl identity, preserving all declared aliases.
template<class F>void tensors(Model& m,F f){
  for(auto& w:m.nodes){for(auto* t:{&w.decay,&w.weight,&w.bias,&w.read})f(*t);for(auto& [name,t]:w.extra)f(t);}
  for(auto& w:m.regions)for(auto& [name,t]:w.extra)f(t);
  for(auto* group:{&m.input_scale,&m.agg_scale,&m.edge_scale,&m.output_scale})for(auto& t:*group)f(t);
}
void locate(Continuation& q,const Graph& g,const Model& m){
  for(auto& [o,s]:q.states){
    if(o.second<0||static_cast<size_t>(o.second)>=m.nodes.size())fail("invalid state owner");
    auto device=m.nodes[o.second].bias.device();s.value=s.value.detach().to(device);
    for(auto& [name,t]:s.slots)t=t.detach().to(device);
  }
  for(auto& [o,h]:q.history){
    if(o.second<0||static_cast<size_t>(o.second)>=m.regions.size())fail("invalid history owner");
    auto device=region_reference(g,m,o.second).device();for(auto& [name,t]:h.tensors)t=t.detach().to(device);
  }
  for(auto& a:q.pending){
    if(a.node<0||static_cast<size_t>(a.node)>=m.nodes.size())fail("invalid pending target");
    a.value=a.value.detach().to(m.nodes[a.node].bias.device());
  }
}
}
void GraphCheckpoint::save(const std::filesystem::path& path,const Graph& g,const Model& m,
    const Continuation& q,const NamedOptimizer* optimizer,int64_t max_file_bytes){
  file_limit(max_file_bytes);
  validate_model(g,m);auto checked=q;validate_window(g,m,checked,{},q.cut,q.cut);
  auto registry=m.parameters(false);GraphRecord record;record.continuation=q;auto& p=record.parameters;p.identity=g.identity;
  for(const auto& o:registry.owners())p.owners.push_back({o.aliases,o.value});
  if(optimizer){p.has_optimizer=true;p.optimizer_class=optimizer->layout().class_name;
    p.groups=optimizer->groups();p.state=optimizer->state();p.precision=optimizer->precision();p.masters=optimizer->masters();}
  validate_decoded(p,registry,optimizer,g.identity);
  publish_exclusive(path,encode_graph(record,max_file_bytes));
}
Continuation GraphCheckpoint::load(const std::filesystem::path& path,const Graph& g,Model& m,NamedOptimizer* optimizer,int64_t max_file_bytes){
  auto record=decode_graph(read_all(path,max_file_bytes));auto registry=m.parameters(false);auto& p=record.parameters;
  validate_decoded(p,registry,optimizer,g.identity);
  // Allocate every transfer and validate candidate values, state, and ownership
  // before publishing any live parameter or optimizer change.
  at::NoGradGuard guard;
  auto current=owner_values(registry);std::map<const void*,Tensor> future;
  for(auto& o:p.owners){const auto& target=current.at(o.aliases.front());
    o.value=o.value.to(target.device());future[target.unsafeGetTensorImpl()]=o.value.set_requires_grad(target.requires_grad());}
  auto staged=m;tensors(staged,[&](Tensor& t){t=future.at(t.unsafeGetTensorImpl());});
  validate_model(g,staged);locate(record.continuation,g,staged);
  validate_window(g,staged,record.continuation,{},record.continuation.cut,record.continuation.cut);
  for(auto& [name,s]:p.state){auto device=current.at(name).device();
    for(auto* t:{&s.momentum_buffer,&s.exp_avg,&s.exp_avg_sq,&s.max_exp_avg_sq})if(t->defined())*t=t->to(device);}
  for(auto& [name,t]:p.masters)t=t.to(current.at(name).device());
  for(const auto& o:p.owners)current.at(o.aliases.front()).copy_(o.value);
  if(optimizer){optimizer->groups_.swap(p.groups);optimizer->state_.swap(p.state);
    optimizer->masters_.swap(p.masters);optimizer->precision_=p.precision;optimizer->scaled_backward_=false;}
  return std::move(record.continuation);
}
}
