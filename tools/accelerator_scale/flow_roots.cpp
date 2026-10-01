#include "flow_roots.h"
#include <torch/csrc/autograd/autograd.h>
#include <set>
#include <iostream>

namespace accelerator_scale::flows {
namespace {
struct Root { std::string name; Tensor expected,actual,dependencies; };
void check_one(const Root& r,const bounded::Program* program,const std::vector<Tensor>& expected_leaves,
               const std::vector<Tensor>& leaves,const std::vector<Tensor>& external,double rtol,double atol) {
  for(Index direction=0;direction<3;++direction) {
    auto index=at::arange(r.expected.numel(),at::TensorOptions().dtype(at::kLong)).reshape(r.expected.sizes());
    auto u=((index*(direction+1)+1).remainder(7+direction*4)-(3+direction*2)).to(r.expected.scalar_type())/8.;
    if(direction==2)u=at::zeros_like(u);
    std::vector<Tensor> eg(expected_leaves.size()),ag(leaves.size());
    if(r.expected.requires_grad())eg=torch::autograd::grad({r.expected},expected_leaves,{u},true,false,true);
    if(program) {
      auto root=bounded::root(r.actual,r.dependencies);
      ag=bounded::export_gradients(root,program->vjp(root,u.to(root.data.device()),true,external));
    } else if(r.actual.requires_grad())ag=torch::autograd::grad({r.actual},leaves,{u.to(r.actual.device())},true,false,true);
    for(size_t j=0;j<eg.size();++j) {
      if(eg[j].defined()!=ag[j].defined())throw std::runtime_error("isolated None/zero mismatch "+r.name+" owner="+std::to_string(j));
      if(!eg[j].defined())continue;
      auto a=ag[j].detach().to(at::kCPU).to(at::kDouble),b=eg[j].detach().to(at::kCPU).to(at::kDouble);
      if(!at::isfinite(a).all().item<bool>() || !at::allclose(a,b,rtol,atol))
        throw std::runtime_error("isolated VJP mismatch "+r.name+" owner="+std::to_string(j)+" max="+std::to_string((a-b).abs().max().item<double>()));
    }
  }
}
}
void check_roots(const Result& expected,const Result& actual,const bounded::Program* program,const bounded::Window& window,
                 const std::vector<Tensor>& expected_leaves,const std::vector<Tensor>& leaves,
                 const std::vector<Tensor>& external,double rtol,double atol) {
  std::vector<Root> roots;std::set<std::pair<bool,bool>> chosen;
  for(size_t i=0;i<expected.trace.size();++i) {
    const auto& e=expected.trace[i];if(e.batch!=0)continue;
    // Active/inactive body and identity boundary roots, independent of outputs.
    const bool boundary=program?program->graph().nodes[e.node].identity:!e.descriptor.requires_grad();
    if(!chosen.emplace(boundary,e.active).second)continue;
    const auto& a=actual.trace.at(i);const bounded::Event* be=nullptr;
    if(program)for(const auto& v:window.events)if(v.node==e.node && v.time==e.time){be=&v;break;}
    if(program && !be)throw std::logic_error("missing finite event root");
    auto add=[&](std::string name,const Tensor& er,const Tensor& ar,const bounded::Value* v) {
      if(!er.defined())return;
      roots.push_back({name+" node="+std::to_string(e.node),er,program?v->data[e.batch]:ar,
        program?v->dependencies[e.batch].unsqueeze(0):Tensor()});
    };
    add("content",e.content,a.content,be?&be->content:nullptr);
    add("proposal",e.proposal,a.proposal,be?&be->proposal.value:nullptr);
    add("next",e.next,a.next,be?&be->next.value:nullptr);
    add("Read",e.descriptor,a.descriptor,be?&be->descriptor:nullptr);
    add("control",e.control,a.control,be?&be->control:nullptr);
    if(e.active) {
      add("Full",e.full,a.full,be?&be->fresh:nullptr);
      if(!e.emitted.empty())add("Emit",e.emitted[0].value,a.emitted[0].value,be?&be->emitted[0]:nullptr);
    }
  }
  size_t states=0;
  for(const auto& [owner,state]:expected.continuation.states) {
    if(owner.first || !state.value.requires_grad() || states++>=2)continue;
    const auto& a=actual.continuation.states.at(owner);
    if(program) {
      const auto& b=window.states[owner.second];
      roots.push_back({"final state",state.value,b.value.data[0],b.value.dependencies[0].unsqueeze(0)});
      if(b.valid.defined()) {
        auto visible=at::nonzero(b.valid[0]).reshape({-1});
        for(auto [name,v]:std::vector<std::pair<std::string,Tensor>>{{"key",b.key},{"value",b.cache_value},{"log_bias",b.bias}})
          roots.push_back({"cache "+name,state.slots.at(name),v[0].index_select(0,visible),
            name=="log_bias"?program->empty_dependencies(v.device()).slice(0,0,1):b.cache_dependencies[0].unsqueeze(0)});
      }
    } else {
      roots.push_back({"final state",state.value,a.value,{}});
      for(const auto& [name,v]:state.slots)roots.push_back({"cache "+name,v,a.slots.at(name),{}});
    }
  }
  if(roots.empty())throw std::runtime_error("empty isolated complete-flow gate");
  for(const auto& root:roots)check_one(root,program,expected_leaves,leaves,external,rtol,atol);
  std::cout<<"PASS isolated content/proposal/Next/Read/control/Full/Emit/state/cache roots="<<roots.size()<<'\n'<<std::flush;
}
}  // namespace accelerator_scale::flows
