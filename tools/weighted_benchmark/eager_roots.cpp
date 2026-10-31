#include "eager.h"
#include <ATen/core/grad_mode.h>
#include <algorithm>
namespace weighted_benchmark {
namespace {
std::string owner_name(const Owner& o){return std::to_string(o.first)+"."+std::to_string(o.second);}
Continuation leaf_state(Continuation q,std::map<std::string,Tensor>& leaves){
  auto leaf=[&](Tensor& x,const std::string& name){
    if(!x.defined())return;x=x.detach().clone();
    if(x.is_floating_point()){x.set_requires_grad(true);leaves.emplace(name,x);}
  };
  for(auto& [owner,s]:q.states){const auto name="state."+owner_name(owner);
    leaf(s.value,name+".value");for(auto& [field,x]:s.slots)leaf(x,name+".slot."+field);}
  for(auto& [owner,h]:q.history)for(auto& [field,x]:h.tensors)leaf(x,"history."+owner_name(owner)+"."+field);
  std::map<std::string,size_t> occurrences;
  for(auto& x:q.pending){const auto name="pending."+std::to_string(x.batch)+"."+std::to_string(x.node)+"."+
      std::to_string(x.time)+"."+std::to_string(x.kind)+"."+std::to_string(x.source)+"."+std::to_string(x.position);
    leaf(x.value,name+"."+std::to_string(occurrences[name]++));}
  return q;
}
}
RootDiagnostic Eager::diagnostic(bool split)const{
  at::AutoGradMode grad(true);RootDiagnostic out;
  for(const auto& p:registry.owners())out.leaves["parameter."+p.canonical]=p.value;
  Continuation q;q.identity=data.graph.identity;q.batch_size=data.samples;
  auto xs=inputs(data,0);out.stop=data.span;
  if(split){
    // Prefixes are independently executed by each schedule. Their numerical
    // fields become independent suffix leaves; no reference trajectory is fed.
    Index earliest=data.span;for(const auto& x:xs)earliest=std::min(earliest,x.time);
    out.start=std::min(data.span,std::max(data.span/2,earliest+1));
    std::vector<External> head,tail;
    for(auto x:xs)(x.time<out.start?head:tail).push_back(std::move(x));
    {at::NoGradGuard no_grad;q=forward(q,placed_inputs(std::move(head)),out.start).continuation;}
    q=leaf_state(std::move(q),out.leaves);
    if(out.start==data.span){tail=inputs(data,1);out.stop=2*data.span;}
    xs=std::move(tail);
  }
  out.initial=q;out.result=forward(q,placed_inputs(std::move(xs),&out.leaves),out.stop);return out;
}
}  // namespace weighted_benchmark
