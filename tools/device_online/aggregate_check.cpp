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
const std::array<std::string,5> kinds{"sum","mean","weighted_mean","active_softmax","all_softmax"};
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
template<class F> void refuses(F f,const std::string& marker) {
  try{f();}catch(const std::exception& e){if(std::string(e.what()).find(marker)!=std::string::npos)return;throw;}
  throw std::runtime_error("missing Aggregate refusal: "+marker);
}
void coefficients(NodeWeights& w,Index slots,int kind) {
  if(kind<2)return;
  for(Index i=0;i<slots;++i)w.extra[(kind==2?"agg_mass_":"agg_logit_")+std::to_string(i)]=at::full({},.25f*float(i)-.5f,at::kFloat);
}
test::Fixture single(Index width,int kind) {
  test::Fixture f;auto& g=f.graph;g.nodes={{0}};
  g.nodes[0].memory="ema";g.nodes[0].full="identity";g.nodes[0].aggregation=kinds[kind];
  g.regions={{1,true,false,"proposal","count-v1"}};g.inputs={0,0,0,0};g.outputs={0};g.compile();
  g.source_domain->input={2,0,1,0};g.compile();
  NodeWeights w{at::zeros({width},at::kFloat),at::eye(width,at::kFloat),
    at::zeros({width},at::kFloat),at::ones({width},at::kFloat)};
  coefficients(w,3,kind);f.model.nodes={w};
  f.model.input_scale={at::ones({},at::kFloat),at::full({},.5f,at::kFloat),at::full({},2.f,at::kFloat),at::full({},.5f,at::kFloat)};
  f.model.output_scale={at::ones({},at::kFloat)};f.initial.identity=g.identity;f.initial.batch_size=1;return f;
}
ContentLimits limits() {
  ContentLimits l;l.queue=96;l.arrivals=192;l.outputs=256;l.trace=1024;l.workspace_bytes=256*1024*1024;
  l.chunk_policy=ChunkPolicy::aggressive;l.aggregate_chunk_rows=4;return l;
}
Index anchors(at::Device device) {
  Index cases=0;
  for(int kind=0;kind<5;++kind)for(Index width:{1,33,257})for(bool present_zero:{false,true})
  for(bool vectorized:{false,true}) {
    auto f=single(width,kind);auto l=limits();l.vectorized_aggregate=vectorized;l.aggregate_chunk_rows=vectorized?4:1;
    ContentFlow flow(f.graph,f.model,f.initial,device,l);Streaming cpu(f.graph,f.model,{});auto q=f.initial;
    // An exclusive alias changes physical port while preserving the logical
    // denominator. Different inputs/windows are consumed by the same program.
    for(Index t=0;t<2;++t) {
      const Index port=t==0?1:3;
      std::vector<External> xs{{0,port,0,t*3,at::full({width},2.f,at::kFloat)},
        {0,2,t,t*3,at::full({width},1.5f,at::kFloat)}};
      if(present_zero)xs.push_back({0,0,t,t*3,at::zeros({width},at::kFloat)});
      auto expected=cpu.run(q,xs,t*3+1,t*3+1),actual=flow.advance(xs,t*3+1);
      tide_bench::compare(actual,expected,true,at::kFloat);
      double value=4.;
      if(kind==1)value/=present_zero?3.:2.;
      if(kind>=2) {
        std::array<double,3> mass{};double total=0.;
        for(size_t j=0;j<3;++j) {
          const double p=.25*double(j)-.5;mass[j]=kind==2?std::log1p(std::exp(p)):std::exp(p);
          if(j<2||present_zero||kind==4)total+=mass[j];
        }
        value=(mass[0]+3.*mass[1])/total;
      }
      require(at::allclose(actual.trace[0].content,at::full({width},value,at::kFloat),1e-5,1e-6),
        "normalized Aggregate analytic domain/scaling differs");
      require(actual.trace[0].contributions.size()==size_t(present_zero?3:2),"zero source contribution disappeared");
      q=expected.continuation;++cases;
    }
    l.prefill=false;l.vectorized_aggregate=!vectorized;l.diagnostics=false;l.trace=0;
    ContentFlow restored(f.graph,f.model,flow.snapshot(),device,l);restored.advance_device({},8);
    tide_bench::compare(restored.result(),cpu.run(q,{},8,8),false,at::kFloat);++cases;
  }
  return cases;
}
Index windows(at::Device device) {
  Index cases=0;
  const std::array<std::string,3> reads{"content","old","proposal"};
  for(int kind=0;kind<5;++kind)for(int shape=0;shape<4;++shape)for(bool prefill:{false,true}) {
    auto f=test::fixture(shape,1);auto& g=f.graph;
    for(auto& r:g.regions)r.read_mode=reads[(kind+shape)%3];
    for(Index n=0;n<4;++n) {
      const int k=(kind+n)%5;g.nodes[n].aggregation=kinds[k];g.nodes[n].full="tanh";
      coefficients(f.model.nodes[n],g.source_counts[n],k);
    }
    for(size_t e=0;e<g.edges.size();++e)if(e%2)g.origins.push_back({Index(e),7,1});
    g.compile();f.initial.identity=g.identity;auto l=limits();l.prefill=prefill;l.vectorized_aggregate=prefill;
    ContentFlow flow(g,f.model,f.initial,device,l);Streaming cpu(g,f.model,{});auto q=f.initial;Index previous=q.cut;
    for(Index step:{2,6,11}) {
      const Index stop=f.initial.cut+step;std::vector<External> xs;
      for(const auto& x:f.input)if(x.time>=previous&&x.time<stop)xs.push_back(x);
      auto expected=cpu.run(q,xs,stop,stop),actual=flow.advance(xs,stop);
      try{tide_bench::compare(actual,expected,true,at::kFloat);}
      catch(...){std::cerr<<"Aggregate kind="<<kind<<" shape="<<shape<<" prefill="<<prefill<<" step="<<step<<'\n';throw;}
      q=expected.continuation;previous=stop;++cases;
    }
    l.prefill=!prefill;ContentFlow restored(g,f.model,flow.snapshot(),device,l);
    tide_bench::compare(restored.advance({},previous+3),cpu.run(q,{},previous+3,previous+3),true,at::kFloat);++cases;
  }
  return cases;
}
Index boundaries(at::Device device) {
  Index cases=0;
  for(int kind:{2,3,4}) {
    auto f=single(1,kind);const auto prefix=kind==2?"agg_mass_":"agg_logit_";
    for(int j=0;j<3;++j)f.model.nodes[0].extra[prefix+std::to_string(j)].fill_(j==2?1000.f:-1000.f);
    f.input={{0,1,0,0,at::ones({1},at::kFloat)},{0,2,0,0,at::ones({1},at::kFloat)}};
    ContentFlow flow(f.graph,f.model,f.initial,device,limits());Streaming cpu(f.graph,f.model,{});
    if(kind==2) {
      refuses([&]{cpu.run(f.initial,f.input,1,1);},"zero mass");
      refuses([&]{flow.advance(f.input,1);},"code=13");refuses([&]{flow.snapshot();},"failed");
    }else tide_bench::compare(flow.advance(f.input,1),cpu.run(f.initial,f.input,1,1),true,at::kFloat);
    ++cases;
  }
  auto f=single(1,4);f.input={{0,1,0,0,at::zeros({1},at::kFloat)},{0,3,0,0,at::zeros({1},at::kFloat)}};
  ContentFlow flow(f.graph,f.model,f.initial,device,limits());
  refuses([&]{flow.advance(f.input,1);},"code=2");++cases;
  return cases;
}
Index domains(at::Device device) {
  Index cases=0;
  for(int kind:{2,3,4})for(Index width:{1,257}) {
    auto f=single(width,kind);auto& g=f.graph;
    g.inputs.assign(257,0);g.layout.reset();g.source_domain.reset();g.compile();f.initial.identity=g.identity;
    f.model.input_scale.assign(257,at::ones({},at::kFloat));coefficients(f.model.nodes[0],257,kind);
    f.input={{0,0,0,0,at::full({width},.5f,at::kFloat)},
      {0,256,0,0,at::full({width},1.5f,at::kFloat)}};
    Streaming cpu(g,f.model,{});ContentFlow flow(g,f.model,f.initial,device,limits());
    auto actual=flow.advance(f.input,1);tide_bench::compare(actual,cpu.run(f.initial,f.input,1,1),true,at::kFloat);
    require(actual.stats.at("retained_tensor_bytes")<=actual.stats.at("planned_buffer_bytes"),
      "normalized Aggregate buffers exceeded static memory estimate");++cases;
  }
  auto f=single(1,4);auto& g=f.graph;g.inputs.clear();g.layout.reset();g.source_domain.reset();g.compile();
  f.initial.identity=g.identity;f.model.input_scale.clear();f.model.nodes[0].extra.clear();
  Streaming cpu(g,f.model,{});ContentFlow flow(g,f.model,f.initial,device,limits());
  tide_bench::compare(flow.advance({},1),cpu.run(f.initial,{},1,1),true,at::kFloat);++cases;
  return cases;
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("Aggregate gate requires explicit NPU FP32");
    auto device=portable_torch::resolve_device(args);if(device.type()!=c10::DeviceType::PrivateUse1)throw std::invalid_argument("Aggregate gate requires NPU");
    at::set_num_threads(1);at::set_num_interop_threads(1);at::NoGradGuard guard;
    const auto a=anchors(device);std::cout<<"Aggregate anchors/restores="<<a<<" passed\n"<<std::flush;
    const auto b=windows(device);std::cout<<"Aggregate windows/restores="<<b<<" passed\n"<<std::flush;
    const auto c=boundaries(device);
    const auto d=domains(device);
    std::cout<<"device-aggregate: passed anchors/restores="<<a<<" windows/restores="<<b<<" boundaries="<<c<<" domains="<<d<<" scope=FP32_HARD_inference\n";
    runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
