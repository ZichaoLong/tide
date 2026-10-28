// Standalone LibTorch literal formula/VJP anchors, pinned weighted KV/Alloc.
#include "tide/weighted.h"
#include "tide/stream.h"
#include "tide/frontier.h"
#include "tide/read.h"
#include "tide/full.h"
#include <torch/csrc/autograd/autograd.h>
#include <ATen/core/grad_mode.h>
#include <cmath>
#include <stdexcept>

void check_weighted_query_vjp(tide::NodeWeights,tide::State,tide::Index,const at::TensorOptions&);

namespace {
using namespace tide;
void close(const Tensor& a,const Tensor& b){
  if(!at::allclose(a,b,a.scalar_type()==at::kDouble?1e-8:1e-5,a.scalar_type()==at::kDouble?1e-10:1e-6))
    throw std::runtime_error("weighted standalone formula mismatch");
}
}
void check_weighted(const at::TensorOptions& o){
  using namespace tide;using namespace tide::weighted;
  auto t=at::tensor({2.,1.,100.},o).set_requires_grad(true);
  auto mu=at::tensor({1.,1.,0.},o).set_requires_grad(true);
  auto total=at::full({},1.4,o).set_requires_grad(true);
  auto tau=at::full({},2.,o).set_requires_grad(true);
  auto allocation=allocate_quad(t,mu,total,total,tau);
  close(allocation.controls,at::tensor({.95,.45,0.},o));
  auto gradients=torch::autograd::grad({allocation.controls[0]},{t,total,tau});
  close(gradients[0],at::tensor({.25,-.25,0.},o));close(gradients[1],at::full({},.5,o));
  close(gradients[2],at::full({},-.125,o));
  // Independent affine face at a large common score offset.
  auto shifted=at::tensor({10000.5,10000.,10002.,9997.},o).set_requires_grad(true);
  auto caps=at::tensor({1.,1.,.25,1.},o).set_requires_grad(true);
  auto bound=at::full({},.95,o).set_requires_grad(true),temp=at::ones({},o).set_requires_grad(true);
  auto offset=allocate_quad(shifted,caps,bound,bound,temp);
  close(offset.controls,at::tensor({.6,.1,.25,0.},o));
  auto bars=torch::autograd::grad({offset.controls},{shifted,caps,bound,temp},{at::tensor({2.,-1.,4.,7.},o)});
  close(bars[0],at::tensor({1.5,-1.5,0.,0.},o));close(bars[1],at::tensor({0.,0.,3.5,0.},o));
  close(bars[2],at::full({},.5,o));close(bars[3],at::full({},-.75,o));
  KVConfig c;c.width=c.summary_width=c.adapter_rank=c.heads=c.ffn_width=1;c.epsilon=1;c.alpha_ff=0;c.window=5;
  RegionalConfig r;r.k=2;r.region_control_mode="fixed";r.total_fraction=.4;r.allocation_temperature=2;
  r.region_evidence.reset();
  Graph g;g.nodes={{0},{0}};g.regions={{2,true,false,"proposal","weighted-total-test"}};
  for(auto& n:g.nodes){n.memory="weighted-kv-test";n.readout="weighted-read-test";n.emission="weighted-full-test";n.aggregation="weighted-aggregate-test";}
  g.inputs={0,1};g.outputs={0,1};g.compile();
  auto base=at::tensor({2.,3.},o).reshape({2,1}).set_requires_grad(true);
  Model model;
  for(Index i=0;i<2;++i){
    NodeWeights w{at::zeros({2},o),at::zeros({2,2},o),at::zeros({2},o),at::zeros({2},o)};
    w.extra={{"kernel.base",base},{"kernel.adapter_u",at::zeros({2,1},o)},
      {"kernel.adapter_v",at::zeros({1,1},o)},{"kernel.summary",at::full({1,1},4.,o)},
      {"read_program.score.matrix",at::zeros({4,8},o)},{"read_program.score.hidden_bias",at::zeros({4},o)},
      {"read_program.score.vector",at::zeros({4},o)},{"read_program.score.bias",at::full({},2.-i,o).set_requires_grad(true)},
      {"full_program.content.query",at::zeros({1,1},o)},{"full_program.content.out",at::ones({1,1},o)},
      {"full_program.content.gate",at::zeros({1,1},o)},{"full_program.content.up",at::zeros({1,1},o)},
      {"full_program.content.down",at::zeros({1,1},o)}};
    configure_node(w,c,r,SourceClasses{{},{0,0}},0.);model.nodes.push_back(w);
  }
  model.regions.resize(1);configure_region(model.regions[0],r);
  {
    auto snapshot=node_binding(model.nodes[0]);snapshot.sources.validate(g);
    if(!same_configuration(std::get<KVConfig>(snapshot.memory),c)||!same_configuration(snapshot.regional,r)
      ||!same_configuration(region_binding(model.regions[0]),r)||snapshot.aggregate_alpha!=0)
      throw std::runtime_error("weighted binding inspection changed installed configuration");
    std::get<KVConfig>(snapshot.memory).window=999;snapshot.sources.inputs[0]=999;
    if(!same_configuration(std::get<KVConfig>(node_binding(model.nodes[0]).memory),c)
      ||node_binding(model.nodes[0]).sources.inputs[0]!=0)
      throw std::runtime_error("weighted inspection mutates installed configuration");
    auto other=model.nodes[0];auto changed=c;changed.window=6;
    configure_node(other,changed,r,SourceClasses{{},{0,0}},0.);
    auto mismatch=model.nodes[0];mismatch.full_kernel=other.full_kernel;
    bool rejected=false;try{node_binding(mismatch);}catch(const std::invalid_argument&){rejected=true;}
    if(!rejected)throw std::runtime_error("weighted inconsistent Full binding accepted");
  }
  model.input_scale=model.output_scale={at::ones({},o),at::ones({},o)};
  auto x=at::tensor({1.,1.},o).set_requires_grad(true),y=at::tensor({1.,-1.},o).set_requires_grad(true);
  Continuation q;q.identity=g.identity;const Index clock=(Index{1}<<54)+9;q.cut=clock;
  std::vector<External> inputs{{0,0,0,clock,x},{0,1,0,clock,y}};
  auto a=Streaming(g,model,Options{}).run(q,inputs,clock+1,clock+1);
  Options options;options.packed=true;
  auto b=Frontier(g,model,options).run(q,inputs,clock+1,clock+1);
  close(a.outputs[0].value,at::tensor({.95,1+3/std::sqrt(2.)},o));
  close(a.outputs[1].value,at::tensor({.45,-1-3/std::sqrt(2.)},o));
  for(size_t i=0;i<a.outputs.size();++i)close(a.outputs[i].value,b.outputs[i].value);
  auto grad=torch::autograd::grad({a.outputs[0].value[1]-a.outputs[1].value[1]},{base})[0];
  close(grad,at::tensor({0.,std::sqrt(2.)},o).reshape({2,1}));
  auto packed_grad=torch::autograd::grad({b.outputs[0].value[1]-b.outputs[1].value[1]},{base})[0];
  close(packed_grad,at::tensor({0.,std::sqrt(2.)},o).reshape({2,1}));
  if(!b.stats.at("batched_state_events")||b.stats.count("semantic_state_replays")&&b.stats.at("semantic_state_replays"))
    throw std::runtime_error("weighted standalone State VJP fell back to scalar replay");
  const auto& state=a.continuation.states.at({0,0});
  check_weighted_query_vjp(model.nodes[0],state,clock,o);
  if(state.slots.at("times")[0].item<Index>()!=clock)throw std::runtime_error("weighted clock lost int64 precision");
  auto zero=model.nodes[0].kernel->step(model.nodes[0],state,ContentView{at::tensor({0.,5.},o)},clock+20);
  if(!at::equal(zero.slots.at("times"),state.slots.at("times")))throw std::runtime_error("weighted zero receive changed exact state");
  auto mem=memory(model.nodes[0],zero,clock+20,at::ones({1},o),c);
  for(const auto& value:mem)close(value,at::zeros_like(value));
  {
    at::NoGradGuard guard;const auto& w=model.nodes[0];
    auto rows=at::tensor({0.,5.,.5,1.,1.,-1.,0.,9.,.3,1.},o).reshape({5,2});
    PackedSequence batch{rows,{0,3,5},{{0,0},{1,0}},
      {clock+1,clock+2,clock+7,clock+1,clock+4},{}};
    for(Index i=0;i<5;++i)batch.views.push_back(ContentView{rows[i]});
    auto actual=w.kernel->packed_sequence(w,{state,state},batch);
    if(actual.calls!=1||actual.scalar_steps!=0||actual.max_batch!=2)
      throw std::runtime_error("weighted KV did not use joint projections");
    for(size_t owner=0;owner<2;++owner){auto expected=state;
      for(Index j=batch.offsets[owner];j<batch.offsets[owner+1];++j){
        expected=w.kernel->step(w,expected,batch.views[j],batch.times[j]);
        for(const auto& [name,value]:expected.slots)close(value,actual.states[j].slots.at(name));
      }
    }
    if(actual.states[2].slots.at("times").numel()!=1||actual.states[2].slots.at("times")[0].item<Index>()!=clock+7)
      throw std::runtime_error("weighted packed KV window lost original time");
  }
  {
    // Describe/Full joint queries: the populated state, an expired view and
    // Read^0 must differ. Explicit formula avoids using fields_step as oracle.
    at::NoGradGuard guard;auto w=model.nodes[0];
    w.extra["read_program.score.matrix"]=at::zeros({4,8},o);
    w.extra["read_program.score.matrix"][0][2].fill_(1);
    w.extra["read_program.score.vector"]=at::tensor({1.,0.,0.,0.},o);
    auto h=at::tensor({1.,1.},o),control=at::full({},.2,o);
    std::vector<ReadInput> reads{{&state,clock,ContentView{h}},{&state,clock+20,ContentView{h}},
      {nullptr,clock,ContentView{h}},{&state,clock,ContentView{h}}};
    auto descriptors=w.read_kernel->fields_batch(w,reads);
    const double short_value=4/std::sqrt(2.);
    for(size_t i=0;i<reads.size();++i){
      close(descriptors[i].at("score"),at::full({},2+((i==0||i==3)?short_value/(1+std::exp(-short_value)):0),o));
      close(descriptors[i].at("w"),at::ones({},o));close(descriptors[i].at("mu"),at::ones({},o));
    }
    std::vector<FullInput> fulls{{&state,clock,ContentView{h},control},{&state,clock+20,ContentView{h},control}};
    auto values=w.full_kernel->batch(w,fulls,3,Options{});
    close(values[0].value,at::tensor({.2,1+3/std::sqrt(2.)},o));
    close(values[1].value,at::tensor({.2,1.},o));
    for(const auto& result:values)for(const auto& emitted:result.emitted)
      if(!emitted.value.is_same(result.value))throw std::runtime_error("weighted Full batch lost broadcast alias");
    if(!w.read_kernel->joint_batch()||!w.full_kernel->joint_batch())throw std::runtime_error("missing weighted query batch capability");
  }
}
