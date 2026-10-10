#include "device_backend.h"
#include "content_fixture.h"
#include "portable_torch/runtime.hpp"
#include "tide/stream.h"
#include "../../cpp/bench/streaming.h"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
using namespace tide;
using namespace tide::device_online;
const std::array<std::string,5> kinds{"sum","mean","linear","active-softmax","all-softmax"};
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void weights(NodeWeights& w,Index width,Index slots,int kind) {
  auto eye=at::eye(width,at::kFloat);
  w.extra["fiber_qkv"]=at::cat({eye*.25f,eye*.5f,eye},1);
  w.extra["fiber_qkv_bias"]=at::zeros({3*width},at::kFloat);
  w.extra["fiber_out"]=eye;w.extra["fiber_out_bias"]=at::zeros({width},at::kFloat);
  w.extra["fiber_decay"]=at::full({},.1f,at::kFloat);
  if(kind>=2)w.extra["fiber_pool"]=at::arange(slots,at::kFloat)*.125f+.125f;
}
State initial(Index width,Index heads,Index rows,Index time) {
  return {at::full({width},.25f,at::kFloat),time,(Index(1)<<55)+5,
    {{"key",at::zeros({rows,heads,width/heads},at::kFloat)},
     {"value",at::full({rows,heads,width/heads},5.f,at::kFloat)},
     {"log_bias",at::full({rows},.125f,at::kFloat)}}};
}
ContentLimits limits() {
  ContentLimits l;l.queue=96;l.arrivals=192;l.outputs=256;l.trace=1024;
  l.kv_rows=64;l.kv_trace_rows=16384;l.attention_chunk_rows=4;l.attention_key_rows=7;
  l.workspace_bytes=256*1024*1024;l.chunk_policy=ChunkPolicy::aggressive;return l;
}
test::Fixture single(Index width) {
  test::Fixture f;auto& g=f.graph;g.nodes={{0}};auto& n=g.nodes[0];
  n.memory="lh-fiber-attention-sum-repeat-v1";n.full="identity";
  n.query_heads=n.kv_heads=width==33?3:1;
  g.regions={{1,true,false,"proposal","count-v1"}};g.inputs={0,0,0};g.outputs={0};g.compile();
  g.layout->input={2,0,1};g.source_domain->input={2,0,1};g.compile();
  NodeWeights w{at::zeros({width},at::kFloat),at::eye(width,at::kFloat),
    at::zeros({width},at::kFloat),at::ones({width},at::kFloat)};
  weights(w,width,3,0);f.model.nodes={w};
  for(int i=0;i<3;++i)f.model.input_scale.push_back(at::ones({},at::kFloat));
  f.model.output_scale={at::ones({},at::kFloat)};f.initial.identity=g.identity;f.initial.batch_size=1;return f;
}
Index anchors(at::Device device) {
  Index cases=0;
  for(Index width:{1,33,257})for(Index chunk:{1,4})for(bool tiled:{false,true})
  for(bool periodic:{false,true})for(Index cached:{0,257}) {
    auto f=single(width);auto& g=f.graph;auto& w=f.model.nodes[0];
    w.extra["fiber_qkv"].narrow(1,0,2*width).zero_();
    const float rate=periodic?.1f:0.f;w.extra["fiber_decay"].fill_(rate);
    if(periodic)g.nodes[0].state_clock={5,1,3};g.compile();f.initial.identity=g.identity;
    f.initial.states[{0,0}]=initial(width,g.nodes[0].query_heads,cached,-1);
    const std::array<Index,4> times=periodic?std::array<Index,4>{1,3,6,8}:std::array<Index,4>{0,1,4,9};
    const std::array<Index,4> local=periodic?std::array<Index,4>{0,2,3,5}:times;
    const std::array<std::vector<Index>,4> ports{{{0,2},{1},{0,1,2},{2,0}}};
    std::array<Index,3> positions{};std::vector<float> values(cached,5.f),bias(cached,.125f);
    std::vector<float> proposals;std::vector<at::Tensor> biases;Index last=-1;
    for(Index i=0;i<4;++i) {
      // Independent scalar oracle: repeated FP32 subtraction, then every key
      // in this fiber is visible to every current query. No triangular mask.
      for(auto& b:bias)for(Index tick=last;tick<local[i];++tick)b-=rate;
      for(auto port:ports[i]) {
        const float value=i==2&&port==1?0.f:float(i+port+1)*.25f;
        f.input.push_back({0,port,positions[port]++,times[i],at::full({width},value,at::kFloat)});
        values.push_back(value);bias.push_back(0.f);
      }
      double numerator=0,denominator=0;
      for(size_t k=0;k<values.size();++k){const auto mass=std::exp(double(bias[k]));numerator+=mass*values[k];denominator+=mass;}
      proposals.push_back(float(ports[i].size()*numerator/denominator));
      biases.push_back(at::tensor(bias));last=local[i];
    }
    auto l=limits();l.queue=l.arrivals=l.outputs=16;l.trace=32;l.kv_trace_rows=2304;
    l.kv_rows=cached+12;l.attention_chunk_rows=chunk;l.attention_key_rows=tiled?7:300;
    ContentFlow flow(g,f.model,f.initial,device,l);Streaming cpu(g,f.model,{});
    auto expected=cpu.run(f.initial,f.input,10,10),actual=flow.advance(f.input,10);
    try{tide_bench::compare(actual,expected,true,at::kFloat);}
    catch(...){std::cerr<<"fiber batch width="<<width<<" chunk="<<chunk<<" tiled="<<tiled<<" periodic="<<periodic<<" cached="<<cached<<'\n';throw;}
    require(actual.stats.at("device_stages")==1&&actual.stats.at("max_node_time_batch")==4,
      "fiber device did not form an actual node-time batch");
    require(actual.trace.size()==4,"fiber analytic event count");
    for(size_t i=0;i<4;++i) {
      require(at::allclose(actual.trace[i].proposal,at::full({width},proposals[i],at::kFloat),1e-5,1e-6),
        "fiber time batching changed complete-fiber visibility/normalization");
      require(at::equal(actual.trace[i].proposed_state.slots.at("log_bias"),biases[i]),
        "fiber time batching changed repeated decay or intermediate bias");
    }
    ++cases;
    l.prefill=false;l.attention_chunk_rows=chunk==1?4:1;l.attention_key_rows=tiled?300:7;
    l.diagnostics=false;l.trace=0;l.kv_trace_rows=0;
    ContentFlow restored(g,f.model,flow.snapshot(),device,l);
    const Index time=periodic?13:15;
    std::vector<External> next{{0,0,positions[0],time,at::full({width},2.f,at::kFloat)}};
    restored.advance_device(next,time+1);
    tide_bench::compare(restored.result(),cpu.run(expected.continuation,next,time+1,time+1),false,at::kFloat);++cases;
  }
  return cases;
}
Index windows(at::Device device) {
  Index cases=0,multi=0;
  const std::array<std::string,3> reads{"content","old","proposal"};
  for(int kind=0;kind<5;++kind)for(int shape=0;shape<4;++shape)for(int policy=0;policy<3;++policy)
  for(bool prefill:{false,true}) {
    auto f=test::fixture(shape,1);auto& g=f.graph;
    const auto mode=reads[(kind+shape+policy)%3];
    for(auto& r:g.regions){r.observe_all=policy!=1;r.read_mode=mode;r.selector="count-v1";}
    for(Index n=0;n<4;++n) {
      const int k=(kind+n)%5;auto& node=g.nodes[n];
      node.memory="lh-fiber-attention-"+kinds[k]+"-repeat-v1";node.full="tanh";
      node.query_heads=node.kv_heads=n%2?3:1;node.clear=policy==2&&n%2==0;
      weights(f.model.nodes[n],3,g.source_counts[n],k);
    }
    f.initial.states[{0,0}]=initial(3,1,2,f.initial.cut-1);g.compile();f.initial.identity=g.identity;
    auto l=limits();l.prefill=prefill;ContentFlow flow(g,f.model,f.initial,device,l);Streaming cpu(g,f.model,{});
    auto q=f.initial;Index previous=q.cut;
    for(Index step:{2,6,11}) {
      const auto stop=f.initial.cut+step;std::vector<External> xs;
      for(const auto& x:f.input)if(x.time>=previous&&x.time<stop)xs.push_back(x);
      auto expected=cpu.run(q,xs,stop,stop),actual=flow.advance(xs,stop);
      try{tide_bench::compare(actual,expected,true,at::kFloat);}
      catch(...){std::cerr<<"fiber batch kind="<<kind<<" shape="<<shape<<" policy="<<policy<<" Read="<<mode<<" prefill="<<prefill<<" step="<<step<<'\n';throw;}
      require(actual.stats.at("max_causal_node_time_batch")<=1,"selection-dependent fiber cache escaped causal fallback");
      if(prefill&&policy==0&&actual.stats.at("max_node_time_batch")>1)++multi;
      q=expected.continuation;previous=stop;++cases;
    }
    l.prefill=!prefill;ContentFlow restored(g,f.model,flow.snapshot(),device,l);
    tide_bench::compare(restored.advance({},previous+3),cpu.run(q,{},previous+3,previous+3),true,at::kFloat);++cases;
  }
  require(multi>0,"general-topology fiber time batching was not exercised");
  std::cout<<"fiber node-time multi-windows="<<multi<<'\n';return cases;
}
Index saturation(at::Device device) {
  Index cases=0;
  for(Index tile:{1,3,16}) {
    auto f=single(1);f.model.nodes[0].extra["fiber_qkv"].narrow(1,0,2).zero_();
    f.model.nodes[0].extra["fiber_decay"].fill_(2e38f);f.initial.states[{0,0}]=initial(1,1,4,-1);
    f.input={{0,0,0,1,at::ones({1},at::kFloat)},{0,2,0,1,at::zeros({1},at::kFloat)},
      {0,0,1,4,at::full({1},2.f,at::kFloat)}};
    auto l=limits();l.kv_rows=16;l.attention_key_rows=tile;
    ContentFlow flow(f.graph,f.model,f.initial,device,l);Streaming cpu(f.graph,f.model,{});
    auto actual=flow.advance(f.input,5),expected=cpu.run(f.initial,f.input,5,5);
    require(actual.stats.at("max_node_time_batch")==2,"saturation did not exercise node-time batching");
    // Only these declared exceptional slots are neutralized on result copies;
    // first check their exact values, including every intermediate old state.
    auto checked=actual,reference=expected;
    auto bias_check=[](State& a,State& b) {
      const auto av=a.slots.at("log_bias"),bv=b.slots.at("log_bias");
      require(at::equal(av,bv),"saturated intermediate/final biases differ");
      a.slots["log_bias"]=at::zeros_like(av);b.slots["log_bias"]=at::zeros_like(bv);
    };
    require(checked.trace.size()==reference.trace.size(),"saturated trace size");
    for(size_t i=0;i<checked.trace.size();++i) {
      auto& a=checked.trace[i];auto& b=reference.trace[i];
      bias_check(a.old,b.old);bias_check(a.proposed_state,b.proposed_state);
      bias_check(a.comparison_state,b.comparison_state);bias_check(a.next_state,b.next_state);
    }
    bias_check(checked.continuation.states.at({0,0}),reference.continuation.states.at({0,0}));
    tide_bench::compare(checked,reference,true,at::kFloat);
    require(actual.trace[0].proposal.item<float>()==1.f&&actual.trace[1].proposal.item<float>()==2.f,
      "saturated old keys poisoned a later finite fiber");++cases;
  }
  return cases;
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("fiber batch gate requires explicit NPU FP32");
    auto device=portable_torch::resolve_device(args);if(device.type()!=tide::device_online::resident_device_type)throw std::invalid_argument("fiber batch gate requires NPU");
    at::set_num_threads(1);at::set_num_interop_threads(1);at::NoGradGuard guard;
    const auto a=anchors(device);std::cout<<"fiber batch anchors/restores="<<a<<" passed\n"<<std::flush;
    const auto b=windows(device);std::cout<<"fiber batch windows/restores="<<b<<" passed\n"<<std::flush;
    const auto c=saturation(device);
    std::cout<<"device-fiber-batch: passed anchors/restores="<<a<<" windows/restores="<<b<<" saturation="<<c<<" scope=FP32_HARD_inference\n";
    runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
