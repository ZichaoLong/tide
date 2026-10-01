#pragma once
#include "content_flow.h"

namespace tide::device_online::test {
struct Fixture {Graph graph;Model model;Continuation initial;std::vector<External> input;};
inline void model_dtype(Model& model,at::ScalarType dtype) {
  for(auto& w:model.nodes) {
    for(auto x:{&w.decay,&w.weight,&w.bias,&w.read})*x=x->to(dtype);
    for(auto& [_,x]:w.extra)x=x.to(dtype);
  }
  for(auto group:{&model.input_scale,&model.agg_scale,&model.edge_scale,&model.output_scale})
    for(auto& x:*group)x=x.to(dtype);
}
inline Fixture fixture(int shape,int variant) {
  Fixture f;auto& g=f.graph;g.nodes={{0},{1},{0},{1}};
  g.regions={{1,variant==0,variant==0,"content","count-v1"},{1,true,true,"content","positive-v1"}};
  const std::vector<std::vector<Edge>> edges{
    {{0,1,2},{0,1,2},{2,3,1},{1,2,3},{3,0,2}},
    {{2,1,3},{2,1,1},{0,3,2},{1,3,4}},
    {},{{0,0,2},{1,1,1},{2,0,3},{0,2,1},{3,3,4}}};
  g.edges=edges[shape];g.inputs={0,2,3};g.outputs={2,1,2};
  for(size_t i=0;i<g.nodes.size();++i) {
    auto& node=g.nodes[i];node.memory=i==2?"identity":"ema";node.full="identity";node.clear=(i%2==0);
    node.identity=i==1&&variant==0;
    NodeWeights w{at::zeros({3},at::kFloat),at::eye(3,at::kFloat),at::zeros({3},at::kFloat),at::tensor({1.f,.5f,-.25f})};
    f.model.nodes.push_back(w);
  }
  g.compile();auto scale=[](float value){return at::full({},value,at::kFloat);};
  for(size_t p=0;p<g.inputs.size();++p)f.model.input_scale.push_back(scale(.5f+p*.125f));
  for(size_t e=0;e<g.edges.size();++e){f.model.agg_scale.push_back(scale(.5f));f.model.edge_scale.push_back(scale(e%2?.25f:.5f));}
  for(size_t p=0;p<g.outputs.size();++p)f.model.output_scale.push_back(scale(p%2?.5f:-.25f));
  const Index base=variant?(Index(1)<<55)+17:0;
  auto& q=f.initial;q.identity=g.identity;q.batch_size=2;q.cut=base;
  q.states[{0,0}]={at::tensor({.25f,-.5f,.125f}),base-1,(Index(1)<<55)+5};
  History history;history.last_time=base-1;history.node_maps["selected"]={{0,0},{2,(Index(1)<<55)+1}};q.history[{0,0}]=history;
  for(Index b=0;b<2;++b)for(Index p=0;p<3;++p)for(Index t=0;t<(b==0?3:2);++t) {
    auto value=at::tensor({float(1+p+t),float(b-p),float(t-b)})*.125f;
    if(variant)value=-value;
    if(p==2&&t==1)value.zero_(); // present zero still creates a candidate.
    f.input.push_back({b,p,t,base+(t==2?5:t),value});
  }
  return f;
}
} // namespace tide::device_online::test
