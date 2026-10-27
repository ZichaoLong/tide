#pragma once
// A public one-node (c,G)=(m,z) binding, with no hidden scaling.
#include <tide/weighted.h>
struct WeightedConsumerFixture {tide::Graph graph;tide::Model model;tide::Continuation initial;};
inline WeightedConsumerFixture weighted_consumer_fixture(){
  using namespace tide;Graph g;g.nodes={{0}};g.regions={{1,true,false,"proposal","weighted-consumer"}};g.inputs=g.outputs={0};
  auto& n=g.nodes[0];n.memory=n.readout=n.emission=n.aggregation="weighted-consumer";g.compile();
  weighted::KVConfig c;c.width=c.summary_width=c.adapter_rank=c.heads=c.ffn_width=1;c.alpha_att=c.alpha_ff=0;
  weighted::RegionalConfig r;r.hidden_width=1;r.region_control_mode="fixed";r.region_evidence.reset();r.total_fraction=.5;
  NodeWeights w{at::zeros({2}),at::zeros({2,2}),at::zeros({2}),at::zeros({2})};
  w.extra={{"kernel.base",at::zeros({2,1})},{"kernel.adapter_u",at::zeros({2,1})},{"kernel.adapter_v",at::zeros({1,1})},
    {"kernel.summary",at::ones({1,1})},{"read_program.score.matrix",at::zeros({1,8})},{"read_program.score.hidden_bias",at::zeros({1})},
    {"read_program.score.vector",at::zeros({1})},{"read_program.score.bias",at::zeros({})}};
  for(const auto& name:{"query","out","gate","up","down"})w.extra["full_program.content."+std::string(name)]=at::zeros({1,1});
  weighted::configure_node(w,c,r,SourceClasses{{},{0}},0);Model m;m.nodes={w};m.regions.resize(1);weighted::configure_region(m.regions[0],r);
  m.input_scale=m.output_scale={at::ones({})};Continuation q;q.identity=g.identity;q.batch_size=1;
  return {g,m,q};
}
