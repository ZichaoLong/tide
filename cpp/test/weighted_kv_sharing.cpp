// KV §1.2 / FC §6.2: shared base and private State-VJP users, literal Jacobian.
#include "tide/state_preparation.h"
#include "tide/state_evaluate.h"
#include "tide/weighted.h"
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/function.h>
#include <cmath>
#include <stdexcept>
#include <utility>
void check_weighted_kv_sharing(const at::TensorOptions& o){
  using namespace tide;using namespace tide::weighted;
  auto close=[&](const Tensor& a,const Tensor& b){
    if(!at::allclose(a,b,o.dtype()==at::kDouble?1e-8:1e-5,o.dtype()==at::kDouble?1e-10:1e-6))
      throw std::runtime_error("shared KV formula/Jacobian mismatch");
  };
  auto base=at::tensor({2.,3.},o).reshape({2,1}).set_requires_grad(true);
  Model m;
  for(Index i=0;i<2;++i){
    NodeWeights w{at::zeros({2},o),at::zeros({2,2},o),at::zeros({2},o),at::zeros({2},o)};
    KVConfig c;c.width=c.summary_width=c.adapter_rank=1;c.epsilon=1;
    configure_node(w,c,RegionalConfig{},SourceClasses{{},{0}},0.);
    w.extra["kernel.base"]=base;
    w.extra["kernel.adapter_u"]=at::tensor({.4,-.2},o).reshape({2,1}).set_requires_grad(true);
    w.extra["kernel.adapter_v"]=at::full({1,1},.5,o).set_requires_grad(true);
    w.extra["kernel.summary"]=at::full({1,1},.7,o).set_requires_grad(true);
    m.nodes.push_back(w);
  }
  auto h=at::tensor({.4,2.},o).set_requires_grad(true);
  std::vector<Event> events(2);
  for(Index i=0;i<2;++i){events[i].node=i;events[i].time=0;events[i].batch=0;events[i].content=h;
    events[i].old=m.nodes[i].kernel->initial(m.nodes[i]);}
  std::map<std::string,Index> stats;prepare_shared_state(m,events,stats);
  if(stats["state_preparation_calls"]!=1||stats["state_preparation_reused_events"]!=1)
    throw std::runtime_error("shared KV repeated base projection");
  if(stats["max_state_private_projection_batch"]!=2||stats["state_private_projection_calls"]!=3)
    throw std::runtime_error("private KV projections were not real row batches");
  for(size_t i=0;i<2;++i)evaluate_state(m,events,{i},true);
  auto& a=events[0].proposed_state;auto& b=events[1].proposed_state;
  const double n=2/std::sqrt(5);
  close(a.slots.at("keys"),at::full({1,1},2.2*n,o));
  close(a.slots.at("values"),at::full({1,1},2.9*n,o));
  close(a.slots.at("summaries"),at::full({1,1},.7*n,o));
  auto loss=a.slots.at("keys").sum()+a.slots.at("values").sum()+
            3*(b.slots.at("keys").sum()+b.slots.at("values").sum())+2*a.slots.at("summaries").sum();
  std::vector<Tensor> roots{h,base,m.nodes[0].extra.at("kernel.adapter_u"),m.nodes[1].extra.at("kernel.adapter_u")};
  auto grads=torch::autograd::grad({loss},roots,{},true);
  close(grads[0],at::tensor({0.,(4*(2.2+2.9)+2*.7)/std::pow(5.,1.5)},o));
  close(grads[1],at::full_like(base,4*n));
  close(grads[2],at::full_like(roots[2],.5*n));close(grads[3],at::full_like(roots[3],1.5*n));
  auto unused=torch::autograd::grad({a.slots.at("keys").sum()},{roots[3]},{},false,false,true);
  if(unused[0].defined())throw std::runtime_error("shared KV merged private adapter roots");
  events[1].content=h.detach().clone().set_requires_grad(true);events[0].state_prepared.clear();events[1].state_prepared.clear();stats.clear();
  prepare_shared_state(m,events,stats);
  if(!events[0].state_prepared.empty()||!events[1].state_prepared.empty())
    throw std::runtime_error("shared KV merged independent equal content roots");
  // Opaque identity keys test equality, not execution order. Make their
  // lexical order explicitly opposite to first-use order, independent of
  // allocator addresses. A restored owner must create its graph consistently.
  auto first=h.detach().clone().set_requires_grad(true),second=first.detach().clone().set_requires_grad(true);
  const auto& w=m.nodes[0];
  if(w.kernel->preparation_key(w,first)<w.kernel->preparation_key(w,second))std::swap(first,second);
  std::vector<Event> interleaved(4);
  for(size_t i=0;i<4;++i){interleaved[i].node=i/2;interleaved[i].content=i%2?second:first;}
  stats.clear();prepare_shared_state(m,interleaved,stats);
  const auto& a0=interleaved[0].state_prepared.at("values");
  const auto& b0=interleaved[1].state_prepared.at("values");
  if(stats["state_preparation_calls"]!=2||stats["state_preparation_events"]!=4||
      a0.grad_fn()->sequence_nr()>=b0.grad_fn()->sequence_nr())
    throw std::runtime_error("shared preparation followed opaque addresses instead of first use");
}
