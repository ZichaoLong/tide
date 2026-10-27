// FC §6.2 / architecture §2: actual reuse, physical sources, all user adjoints.
#include "tide/aggregate_sharing.h"
#include "tide/weighted.h"
#include <torch/csrc/autograd/autograd.h>
#include <stdexcept>
void check_weighted_sharing(const at::TensorOptions& o){
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
