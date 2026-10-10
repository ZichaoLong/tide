#include "device_backend.h"
#include "content_fixture.h"
#include "portable_torch/runtime.hpp"
#include "tide/stream.h"
#include "tide/greedy.h"
#include "../../cpp/bench/streaming.h"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using namespace tide;
using namespace tide::device_online;
const std::array<std::string,5> kinds{"sum","mean","linear","active-softmax","all-softmax"};
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
template<class F> void refuses(F f,const std::string& marker) {
  try{f();}catch(const std::exception& e){if(std::string(e.what()).find(marker)!=std::string::npos)return;throw;}
  throw std::runtime_error("missing fiber pool refusal: "+marker);
}
std::string profile(int kind){return "lh-fiber-attention-"+kinds.at(kind)+"-repeat-v1";}
ContentLimits limits(Index chunk=4) {
  ContentLimits l;l.queue=96;l.arrivals=192;l.outputs=192;l.trace=1024;
  l.kv_rows=32;l.kv_trace_rows=8192;l.attention_chunk_rows=chunk;l.workspace_bytes=256*1024*1024;
  l.attention_key_rows=chunk==1?1:7;
  return l;
}
void weights(NodeWeights& w,Index width,Index slots,int kind) {
  auto eye=at::eye(width,at::kFloat);
  w.extra["fiber_qkv"]=at::cat({eye*.25f,eye*.5f,eye},1);
  w.extra["fiber_qkv_bias"]=at::zeros({3*width},at::kFloat);
  w.extra["fiber_out"]=eye*.5f;w.extra["fiber_out_bias"]=at::full({width},.03125f,at::kFloat);
  w.extra["fiber_decay"]=at::full({},.03125f,at::kFloat);
  if(kind>=2)w.extra["fiber_pool"]=at::arange(slots,at::kFloat)*.125f-.125f;
}
State initial(Index width,Index heads,Index time) {
  return {at::full({width},.125f,at::kFloat),time,5,
    {{"key",at::arange(2*width,at::kFloat).reshape({2,heads,width/heads})*.015625f},
     {"value",at::full({2,heads,width/heads},.25f,at::kFloat)},
     {"log_bias",at::tensor({-.0625f,.125f})}}};
}
test::Fixture one_node(Index width,Index slots,int kind) {
  test::Fixture f;auto& g=f.graph;g.nodes={{0}};auto& node=g.nodes[0];
  node.memory=profile(kind);node.full="tanh";node.query_heads=node.kv_heads=width==33?3:1;
  g.regions={{1,true,false,"proposal","count-v1"}};g.inputs.assign(slots,0);g.outputs={0};g.compile();
  NodeWeights w{at::zeros({width},at::kFloat),at::eye(width,at::kFloat)*.5f,
    at::zeros({width},at::kFloat),at::ones({width},at::kFloat)};
  weights(w,width,slots,kind);f.model.nodes={w};
  for(Index i=0;i<slots;++i)f.model.input_scale.push_back(at::ones({},at::kFloat));
  f.model.output_scale={at::ones({},at::kFloat)};f.initial.identity=g.identity;f.initial.batch_size=1;
  return f;
}
Index anchors(at::Device device) {
  Index cases=0;
  for(int kind=0;kind<5;++kind)for(bool present_zero:{false,true})
  for(bool alias:{false,true})for(bool cached:{false,true})for(Index chunk:{1,4}) {
    auto f=one_node(1,3,kind);auto& g=f.graph;auto& w=f.model.nodes[0];
    // Four physical inputs but only three logical sources. The alternative for
    // slot zero must not add an all-softmax denominator term.
    g.inputs.push_back(0);g.layout.reset();g.source_domain.reset();g.compile();
    g.source_domain->input={2,0,1,0};g.compile();f.initial.identity=g.identity;
    f.model.input_scale.push_back(at::ones({},at::kFloat));
    w.extra["fiber_qkv"]=at::ones({1,3},at::kFloat);w.extra["fiber_out"].fill_(2);
    w.extra["fiber_out_bias"].fill_(.25f);w.extra["fiber_decay"].zero_();
    const std::array<double,3> coefficients{0.,-2.,5.},masses{2.,3.,5.};
    if(kind==2)w.extra["fiber_pool"]=at::tensor({0.f,-2.f,5.f});
    if(kind>=3)w.extra["fiber_pool"]=at::tensor({std::log(2.f),std::log(3.f),std::log(5.f)});
    if(cached)f.initial.states[{0,0}]={at::zeros({1},at::kFloat),-1,5,
      {{"key",at::zeros({1,1,1},at::kFloat)},{"value",at::full({1,1,1},5.f,at::kFloat)},
       {"log_bias",at::zeros({1},at::kFloat)}}};
    f.input={{0,alias?3:1,0,0,at::ones({1},at::kFloat)},
      {0,2,0,0,at::full({1},3.f,at::kFloat)}};
    if(present_zero)f.input.push_back({0,0,0,0,at::zeros({1},at::kFloat)});
    Streaming cpu(g,f.model,{});const auto expected=cpu.run(f.initial,f.input,1,1);
    ContentFlow flow(g,f.model,f.initial,device,limits(chunk));const auto actual=flow.advance(f.input,1);
    tide_bench::compare(actual,expected,true,at::kFloat);
    const std::vector<double> xs=present_zero?std::vector<double>{1.,3.,0.}:std::vector<double>{1.,3.};
    double pooled=0.;
    for(size_t i=0;i<xs.size();++i) {
      // Independent scalar attention and pooling; no core pooling helper.
      double denominator=cached?1.:0.,numerator=cached?5.:0.;
      for(double x:xs){const auto mass=std::exp(xs[i]*x);denominator+=mass;numerator+=mass*x;}
      const double coefficient=kind==0?1.:kind==1?1./xs.size():kind==2?coefficients[i]
        :masses[i]/(kind==4||present_zero?10.:5.);
      pooled+=coefficient*numerator/denominator;
    }
    require(std::abs(actual.trace[0].proposal.item<double>()-(2.*pooled+.25))<2e-5,"post-attention pooling analytic anchor failed");
    const auto& kv=actual.continuation.states.at({0,0}).slots;
    std::vector<float> keys;if(cached)keys.push_back(0.f);for(double x:xs)keys.push_back(float(x));
    require(at::equal(kv.at("key").flatten(),at::tensor(keys)),"pooling changed KV rows/order");
    require(kv.at("key").size(0)==Index(xs.size()+cached),"zero coefficient/source dropped a cache row");++cases;
  }
  return cases;
}
Index domains(at::Device device) {
  Index cases=0;
  for(int kind=0;kind<5;++kind)for(Index width:{7,33,257}) {
    auto f=one_node(width,257,kind);auto& w=f.model.nodes[0];
    w.extra["fiber_qkv"].narrow(1,0,2*width).zero_();
    if(kind>=3){w.extra["fiber_pool"].zero_();w.extra["fiber_pool"][128].fill_(2);}
    f.input={{0,0,0,0,at::ones({width},at::kFloat)},
      {0,256,0,0,at::full({width},3.f,at::kFloat)}};
    Streaming cpu(f.graph,f.model,{});const auto expected=cpu.run(f.initial,f.input,1,1);
    ContentFlow flow(f.graph,f.model,f.initial,device,limits(1));auto actual=flow.advance(f.input,1);
    tide_bench::compare(actual,expected,true,at::kFloat);
    if(kind==4)require(std::abs(actual.trace[0].proposal[0].item<double>()-(2./(256+std::exp(2.))+.03125))<1e-6,
      "all-softmax lost missing logical slots");
    ++cases;
  }
  // Missing dominant logits must not underflow active-only normalization.
  for(int kind:{3,4}) {
    auto f=one_node(1,3,kind);auto& w=f.model.nodes[0];
    w.extra["fiber_qkv"].narrow(1,0,2).zero_();w.extra["fiber_pool"]=at::tensor({-1000.f,1000.f,-1000.f});
    f.input={{0,0,0,0,at::ones({1},at::kFloat)},{0,2,0,0,at::full({1},3.f,at::kFloat)}};
    Streaming cpu(f.graph,f.model,{});const auto expected=cpu.run(f.initial,f.input,1,1);
    ContentFlow flow(f.graph,f.model,f.initial,device,limits(4));auto actual=flow.advance(f.input,1);
    tide_bench::compare(actual,expected,true,at::kFloat);
    require(std::abs(actual.trace[0].proposal.item<double>()-(kind==3?1.03125:.03125))<1e-6,
      "softmax used the wrong domain");++cases;
  }
  auto empty=one_node(1,0,4);
  Streaming cpu(empty.graph,empty.model,{});
  ContentFlow flow(empty.graph,empty.model,empty.initial,device,limits());
  tide_bench::compare(flow.advance({},2),cpu.run(empty.initial,{},2,2),true,at::kFloat);++cases;
  return cases;
}
Index refusals(at::Device device) {
  Index cases=0;
  for(int variant=0;variant<3;++variant) {
    auto f=one_node(1,3,4);auto& w=f.model.nodes[0].extra["fiber_pool"];
    if(variant==0)w=at::zeros({2},at::kFloat);
    if(variant==1)w[1].fill_(std::numeric_limits<float>::quiet_NaN());
    if(variant==2)w=w.to(at::kDouble);
    refuses([&]{ContentFlow flow(f.graph,f.model,f.initial,device,limits());},"fiber pooling parameter/domain");++cases;
  }
  auto f=one_node(1,3,4);f.graph.source_domain->input={0,0,1};f.graph.compile();
  f.initial.identity=f.graph.identity;f.model.nodes[0].extra["fiber_pool"]=at::zeros({2},at::kFloat);
  f.input={{0,0,0,0,at::zeros({1},at::kFloat)},{0,1,0,0,at::zeros({1},at::kFloat)}};
  Streaming cpu(f.graph,f.model,{});
  refuses([&]{cpu.run(f.initial,f.input,1,1);},"duplicate logical source");
  ContentFlow flow(f.graph,f.model,f.initial,device,limits());
  refuses([&]{flow.advance(f.input,1);},"code=2");
  refuses([&]{flow.snapshot();},"failed");++cases;
  f=one_node(1,3,2);auto l=limits();l.kv_rows=1;
  f.input={{0,0,0,0,at::zeros({1},at::kFloat)},{0,1,0,0,at::zeros({1},at::kFloat)}};
  ContentFlow small(f.graph,f.model,f.initial,device,l);
  refuses([&]{small.advance(f.input,1);},"code=11");++cases;
  return cases;
}
Index windows(at::Device device) {
  Index cases=0;
  for(int kind=0;kind<5;++kind)for(int shape:{0,1})for(int variant:{0,1})for(bool prefill:{false,true}) {
    auto f=test::fixture(shape,variant);auto& g=f.graph;
    const std::array<std::string,3> reads{"content","old","proposal"};
    for(auto& r:g.regions)r.read_mode=reads[(kind+variant)%3];
    for(Index n:{0,2}) {
      auto& node=g.nodes[n];const int k=(kind+(n==2))%5;
      node.memory=profile(k);node.full="tanh";node.query_heads=node.kv_heads=n==0?1:3;
      node.clear=variant==0;
      if(n==2)node.readout="norm-fp32-v1";
      weights(f.model.nodes[n],3,g.source_counts[n],k);
    }
    f.initial.states[{0,0}]=initial(3,1,f.initial.cut-1);
    f.initial.states[{0,0}].observations=(Index(1)<<55)+5;
    for(size_t e=0;e<g.edges.size();++e)if(e%2)g.origins.push_back({Index(e),7,1});
    g.compile();f.initial.identity=g.identity;
    auto l=limits(prefill?4:1);l.prefill=prefill;
    l.vectorized_read=prefill;l.vectorized_state=prefill;
    ContentFlow flow(g,f.model,f.initial,device,l);Streaming cpu(g,f.model,{});Greedy greedy(g,f.model,{});
    auto q=f.initial;Index previous=q.cut;
    for(Index offset:{2,6,11}) {
      const auto stop=f.initial.cut+offset;std::vector<External> xs;
      for(const auto& x:f.input)if(x.time>=previous&&x.time<stop)xs.push_back(x);
      const auto expected=cpu.run(q,xs,stop,stop);tide_bench::compare(greedy.run(q,xs,stop,stop),expected,true,at::kFloat);
      const auto actual=flow.advance(xs,stop);tide_bench::compare(actual,expected,true,at::kFloat);
      q=expected.continuation;previous=stop;++cases;
    }
    l.prefill=!prefill;l.diagnostics=false;l.trace=0;l.kv_trace_rows=0;
    ContentFlow restored(g,f.model,flow.snapshot(),device,l);restored.advance_device({},previous+3);
    tide_bench::compare(restored.result(),cpu.run(q,{},previous+3,previous+3),false,at::kFloat);++cases;
  }
  return cases;
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("fiber pool gate requires explicit NPU FP32");
    auto device=portable_torch::resolve_device(args);if(device.type()!=tide::device_online::resident_device_type)throw std::invalid_argument("fiber pool gate requires NPU");
    at::set_num_threads(1);at::set_num_interop_threads(1);at::NoGradGuard guard;
    const auto a=anchors(device);std::cout<<"fiber pool anchors="<<a<<" passed\n"<<std::flush;
    const auto b=domains(device);std::cout<<"fiber pool domains="<<b<<" passed\n"<<std::flush;
    const auto c=windows(device);
    std::cout<<"fiber pool windows="<<c<<" passed\n"<<std::flush;
    const auto d=refusals(device);
    std::cout<<"device-fiber-pool: passed anchors="<<a<<" domains="<<b<<" windows="<<c<<" refusals="<<d<<" scope=FP32_HARD_inference\n";
    runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
