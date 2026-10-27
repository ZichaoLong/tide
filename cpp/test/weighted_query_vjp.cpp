// KV §2–4 / architecture §3–4: literal local derivatives and independent roots.
#include "tide/weighted.h"
#include "tide/full.h"
#include <torch/csrc/autograd/autograd.h>
#include <ATen/core/grad_mode.h>
#include <cmath>
#include <stdexcept>

namespace {
using namespace tide;
void close(const Tensor& x,const Tensor& y){
  if(!x.defined()||!at::allclose(x,y,x.scalar_type()==at::kDouble?1e-8:1e-5,x.scalar_type()==at::kDouble?1e-10:1e-6))
    throw std::runtime_error("weighted query literal VJP mismatch");
}
}
void check_weighted_query_vjp(tide::NodeWeights w,tide::State state,tide::Index clock,const at::TensorOptions& o){
  using namespace tide;
  state.value=state.value.detach().clone().set_requires_grad(true);
  for(auto& [name,t]:state.slots)t=t.detach().clone().set_requires_grad(t.is_floating_point());
  state.slots["weights"]=at::full({1},.5,o).set_requires_grad(true);
  auto content=at::tensor({.7,1.},o).set_requires_grad(true);
  auto other=at::tensor({.3,-1.},o).set_requires_grad(true);
  auto control=at::full({},.2,o).set_requires_grad(true);
  w.extra["read_program.score.matrix"]=at::zeros({4,8},o);
  w.extra["read_program.score.matrix"][0][2].fill_(1);
  w.extra["read_program.score.vector"]=at::tensor({1.,0.,0.,0.},o);
  std::vector<ReadInput> reads{{&state,clock,ContentView{content}},{&state,clock+20,ContentView{other}},
                             {nullptr,clock,ContentView{other}},{&state,clock,ContentView{other}}};
  auto fields=w.read_kernel->fields_batch_grad(w,reads);
  const double summary=4/std::sqrt(2.),z=.5*summary,sigmoid=1/(1+std::exp(-z));
  const double derivative=sigmoid+z*sigmoid*(1-sigmoid);
  close(fields[0].at("score"),at::full({},2+z*sigmoid,o));
  auto grad=torch::autograd::grad({fields[0].at("score")},
    {state.slots.at("summaries"),state.slots.at("weights"),other,state.value,state.slots.at("keys")},{},true,false,true);
  close(grad[0],at::full({1,1},.5*derivative,o));close(grad[1],at::full({1},summary*derivative,o));
  if(grad[2].defined()||grad[3].defined()||grad[4].defined())throw std::runtime_error("Read VJP connected unused row/state");
  auto zero=torch::autograd::grad({fields[0].at("score")},{state.slots.at("summaries"),other},
    {at::zeros({},o)},true,false,true);
  close(zero[0],at::zeros({1,1},o));
  if(zero[1].defined())throw std::runtime_error("Read VJP materialized an absent row");
  auto second=torch::autograd::grad({fields[3].at("score")},{state.slots.at("summaries")},{},true)[0];
  close(second,at::full({1,1},.5*derivative,o));
  std::vector<FullInput> fulls{{&state,clock,ContentView{content},control},
                             {&state,clock+20,ContentView{other},control}};
  auto outputs=w.full_kernel->batch_grad(w,fulls,3,Options{});
  close(outputs[0].value,at::tensor({.2,1+1.5/std::sqrt(2.)},o));
  auto gradient=torch::autograd::grad({outputs[0].emitted[1].value.sum()},
      {state.slots.at("values"),state.slots.at("weights"),content,other,control},{},true,false,true);
  close(gradient[0],at::full({1,1},.5,o));close(gradient[1],at::full({1},3/std::sqrt(2.),o));
  close(gradient[2],at::tensor({0.,1.},o));close(gradient[4],at::ones({},o));
  if(gradient[3].defined())throw std::runtime_error("Full VJP connected another row input");
  for(const auto& row:outputs)for(const auto& slot:row.emitted)
    if(!slot.value.is_same(row.value))throw std::runtime_error("Full VJP lost broadcast alias");
  bool higher=false,modified=false;
  try{torch::autograd::grad({fields[0].at("score")},{content},{},true,true,true);}
  catch(const std::exception& e){higher=std::string(e.what()).find("first-order")!=std::string::npos;}
  {at::NoGradGuard guard;state.slots.at("summaries").add_(.01);}
  try{torch::autograd::grad({fields[0].at("score")},{content},{},true,false,true);}
  catch(const std::exception& e){modified=std::string(e.what()).find("inplace")!=std::string::npos;}
  if(!higher||!modified)throw std::runtime_error("query VJP failed to enforce saved versions/first order");
}
