// FC §6.2 / architecture §2: actual reuse, physical sources, all user adjoints.
#include "tide/aggregate_sharing.h"
#include "tide/weighted.h"
#include "../src/weighted_row_linear.h"
#include <torch/csrc/autograd/autograd.h>
#include <ATen/core/grad_mode.h>
#include <stdexcept>
namespace {
void check_private_rows(const at::TensorOptions& o){
  using tide::Tensor;using tide::weighted::linear_rows;
  auto x=at::tensor({2.,-1.},o).set_requires_grad(true);
  auto w=at::tensor({3.,4.,-2.,5.},o).reshape({2,2}).set_requires_grad(true);
  auto other=x.detach().clone().set_requires_grad(true),other_w=w.detach().clone().set_requires_grad(true);
  auto ys=linear_rows({x,other},{w,other_w});
  if(!at::equal(ys[0],at::tensor({2.,-9.},o)))throw std::runtime_error("private row literal formula");
  auto gs=torch::autograd::grad({ys[0]},{x,w,other,other_w},{at::tensor({.5,-2.},o)},true,false,true);
  if(!at::equal(gs[0],at::tensor({5.5,-8.},o))||!at::equal(gs[1],at::tensor({1.,-.5,-4.,2.},o).reshape({2,2})))
    throw std::runtime_error("private row independent Jacobian");
  if(gs[2].defined()||gs[3].defined())throw std::runtime_error("private rows connected absent roots");
  auto zero=torch::autograd::grad({ys[0]*0},{x,w,other,other_w},{at::ones_like(ys[0])},true,false,true);
  if(!at::equal(zero[0],at::zeros_like(x))||zero[2].defined()||zero[3].defined())
    throw std::runtime_error("private row zero/None roots differ");
  {at::NoGradGuard guard;w.add_(1);}
  bool rejected=false;try{torch::autograd::grad({ys[0].sum()},{x,w});}catch(const std::exception&){rejected=true;}
  if(!rejected)throw std::runtime_error("private row parameter version not checked");
  auto shared=linear_rows({x,x},{other_w,other_w});
  auto both=torch::autograd::grad({shared[0].sum()+3*shared[1].sum()},{x,other_w});
  if(!at::equal(both[0],at::tensor({4.,36.},o))||!at::equal(both[1],at::tensor({8.,-4.,8.,-4.},o).reshape({2,2})))
    throw std::runtime_error("private row lost shared user contribution");
}
}
void check_weighted_sharing(const at::TensorOptions& o){
  check_private_rows(o);
  using namespace tide;
  Graph g;g.nodes={{0},{0}};g.regions={{2,true,false,"proposal"}};g.inputs={0,1};g.outputs={0,1};g.compile();
  Model m;m.input_scale=m.output_scale={at::ones({},o),at::ones({},o)};
  for(Index i=0;i<2;++i){
    NodeWeights w{at::zeros({2},o),at::zeros({2,2},o),at::zeros({2},o),at::zeros({2},o)};
    weighted::KVConfig c;c.width=c.summary_width=1;
    weighted::configure_node(w,c,weighted::RegionalConfig{},SourceClasses{{},{0,0}},0.);m.nodes.push_back(w);
  }
  auto x=at::tensor({.3,.8},o).set_requires_grad(true);
  std::vector<Event> events(2);std::map<Index,std::vector<size_t>> owners{{0,{0}},{1,{1}}};
  for(Index i=0;i<2;++i){events[i].node=i;events[i].time=0;events[i].batch=0;events[i].fiber={{0,i,0,0,i,0,x}};}
  std::map<std::string,Index> stats;
  auto shared=plan_aggregate_sharing(g,m,events,owners,true,stats);
  if(shared.copies.size()!=1||shared.compute.size()!=1)throw std::runtime_error("identical Aggregate inputs were not shared");
  for(const auto& [node,ids]:shared.compute)evaluate_aggregate(g,m,events,ids,true,false,"batched");
  decode_shared_aggregates(g,m,events,shared);
  if(!events[0].content.is_same(events[1].content)||events[0].sources[0].atom.source==events[1].sources[0].atom.source)
    throw std::runtime_error("shared Aggregate lost actual result or physical source identity");
  auto grad=torch::autograd::grad({events[0].content.sum()+3*events[1].content.sum()},{x})[0];
  if(!at::equal(grad,at::full_like(x,4)))throw std::runtime_error("shared Aggregate lost a user adjoint");
  events[1].fiber[0].value=x.detach().clone().set_requires_grad(true);stats.clear();
  if(!plan_aggregate_sharing(g,m,events,owners,true,stats).copies.empty())
    throw std::runtime_error("Aggregate merged numerically equal independent input roots");
  events[1].fiber[0].value=x;stats.clear();
  if(!plan_aggregate_sharing(g,m,events,owners,false,stats).copies.empty())
    throw std::runtime_error("scalar execution acquired implicit shared batching");
}
