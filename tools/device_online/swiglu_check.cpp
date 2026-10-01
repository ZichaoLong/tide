#include "content_fixture.h"
#include "content_profile.h"
#include "packed_swiglu_full.h"
#include "portable_torch/runtime.hpp"
#include "tide/stream.h"
#include "tide/greedy.h"
#include "../../cpp/bench/streaming.h"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using namespace tide;
using namespace tide::device_online;
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void weights(NodeWeights& w,Index width) {
  auto eye=at::eye(width,at::kFloat);
  w.extra["ffn_gate"]=at::cat({eye*.5f,eye*-.25f},1);
  w.extra["ffn_up"]=at::cat({eye*.25f,eye*.5f},1);
  w.extra["ffn_down"]=at::cat({eye*.5f,eye*.25f},0);
  if(width>1){w.extra["ffn_gate"][0][width-1].fill_(.125f);w.extra["ffn_down"][width][1].fill_(.0625f);}
}
test::Fixture component_fixture(Index width) {
  test::Fixture f;f.graph.nodes={{0},{0},{0}};f.graph.regions={{3}};f.graph.outputs={0,1,2};
  for(Index n=0;n<3;++n) {
    f.graph.nodes[n].memory="identity";f.graph.nodes[n].full=n==1?"identity":"swiglu";
    NodeWeights w{at::zeros({width},at::kFloat),at::eye(width,at::kFloat),at::zeros({width},at::kFloat),at::zeros({width},at::kFloat)};
    if(n!=1)weights(w,width);
    f.model.nodes.push_back(w);f.model.output_scale.push_back(at::ones({},at::kFloat));
  }
  f.graph.compile();f.initial.identity=f.graph.identity;return f;
}
Index components(at::Device device,at::ScalarType dtype) {
  Index cases=0;
  for(Index width:{1,7,33,257})for(Index chunk:{1,4}) {
    auto f=component_fixture(width);test::model_dtype(f.model,dtype);ContentProfile profile(f.graph,f.model,device);
    // Component fault injection after public validation: unused owner must
    // never enter a matmul, activation or multiplication.
    for(auto& [_,x]:profile.model.nodes[2].extra)x.fill_(std::numeric_limits<float>::quiet_NaN());
    PackedSwiGluFull full(profile,device,8,chunk,64*1024*1024);
    auto c=at::zeros({8,4},at::kLong);auto base=(at::arange(8*width,at::kFloat).reshape({8,width}).remainder(17)-8)*.0625f;
    base=base.to(dtype);auto comparison=(base+.125f).to(dtype),content=(base*.5f).to(dtype);
    auto valid=at::zeros({8},at::kBool);
    for(Index i=0;i<8;++i){c[i][1].fill_(i==1?1:(i==3||i==6)?2:0);c[i][2].fill_((Index(1)<<55)+i);c[i][3].copy_(c[i][2]);}
    for(Index i:{0,1,2,5})valid[i].fill_(true);
    c[7][1].fill_(-9);comparison[3].fill_(std::numeric_limits<float>::quiet_NaN());comparison[6].fill_(std::numeric_limits<float>::quiet_NaN());
    ActionBatch input{c.to(device),base.to(device),valid.to(device)};
    auto error=at::zeros({1},c.options().device(device).dtype(at::kInt)),chunks=at::zeros({1},c.options().device(device));
    CannProgram p(device);auto result=full.append_stage(p,input,content.to(device),comparison.to(device),error,chunks);p.finish();
    p.run();require(error.cpu().item<int>()==0,"SwiGLU component refused valid actions");
    auto expected=base.clone();const auto& w=f.model.nodes[0];
    for(Index i:{0,2,5})expected[i].copy_(content[i]+at::matmul(at::silu(at::matmul(comparison[i],w.extra.at("ffn_gate")))
      *at::matmul(comparison[i],w.extra.at("ffn_up")),w.extra.at("ffn_down")));
    require(result.values.scalar_type()==dtype&&at::allclose(result.values.cpu(),expected,dtype==at::kHalf?3e-3:1e-5,dtype==at::kHalf?2e-3:1e-6),"SwiGLU formula/selected-only parity failed");
    for(Index i:{0,2,5}) {
      auto x=comparison[i].to(at::kDouble);
      auto ideal=content[i].to(at::kDouble)+at::matmul(at::silu(at::matmul(x,w.extra.at("ffn_gate").to(at::kDouble)))
        *at::matmul(x,w.extra.at("ffn_up").to(at::kDouble)),w.extra.at("ffn_down").to(at::kDouble));
      require(at::allclose(result.values[i].cpu().to(at::kDouble),ideal,dtype==at::kHalf?3e-3:1e-5,dtype==at::kHalf?2e-3:1e-6),"SwiGLU FP64 formula mismatch");
    }
    require(chunks.cpu().item<Index>()==(3+full.chunk_rows()-1)/full.chunk_rows(),"SwiGLU chunk accounting failed");++cases;
    input.valid.zero_();chunks.zero_();p.run();
    require(at::equal(result.values.cpu(),base)&&chunks.cpu().item<Index>()==0,"empty selection performed SwiGLU");++cases;
  }
  return cases;
}
Index windows(at::Device device) {
  Index cases=0;
  for(int shape:{0,1})for(int variant:{0,1})for(bool prefill:{false,true})for(Index chunk:{1,4})
  for(bool affine:{false,true})for(const std::string read:{"content","proposal"}) {
    auto f=test::fixture(shape,variant);auto& g=f.graph;
    for(auto& r:g.regions)r.read_mode=read;
    for(Index n=0;n<Index(g.nodes.size());++n) {
      auto& node=g.nodes[n];if(node.identity)continue;
      node.full=n%2?(n==1?"tanh":"lh-relu-identity-v1"):"swiglu";
      auto& w=f.model.nodes[n];if(node.full=="swiglu")weights(w,3);
      if(affine) {
        node.emission="slot_affine";node.emit_period=2;
        for(Index s=0;s<g.outgoing_ports.offsets[n+1]-g.outgoing_ports.offsets[n];++s) {
          node.emit_phases.push_back(s%2?-2:-1);
          w.extra["emit_w_"+std::to_string(s)]=at::eye(3,at::kFloat)*.5f;
          w.extra["emit_b_"+std::to_string(s)]=at::full({3},.03125f,at::kFloat);
        }
      }
    }
    g.compile();f.initial.identity=g.identity;
    ContentLimits l;l.queue=128;l.arrivals=256;l.outputs=256;l.trace=2048;l.workspace_bytes=256*1024*1024;
    l.full_chunk_rows=chunk;l.emission_chunk_rows=chunk;l.prefill=prefill;
    ContentFlow flow(g,f.model,f.initial,device,l);Streaming reference(g,f.model,{});Greedy greedy(g,f.model,{});
    auto q=f.initial;Index previous=q.cut;
    for(Index offset:{2,6,11}) {
      const Index stop=f.initial.cut+offset;std::vector<External> xs;
      for(const auto& x:f.input)if(x.time>=previous&&x.time<stop)xs.push_back(x);
      const auto expected=reference.run(q,xs,stop,stop);tide_bench::compare(greedy.run(q,xs,stop,stop),expected,true,at::kFloat);
      const auto actual=flow.advance(xs,stop);tide_bench::compare(actual,expected,true,at::kFloat);
      require(actual.stats.at("swiglu_full_chunk_rows")==chunk,"SwiGLU effective chunk missing");q=expected.continuation;previous=stop;++cases;
    }
    l.prefill=!prefill;l.diagnostics=false;l.trace=0;
    ContentFlow restored(g,f.model,flow.snapshot(),device,l);
    tide_bench::compare(restored.advance({},previous+3),reference.run(q,{},previous+3,previous+3),false,at::kFloat);++cases;
  }
  return cases;
}
Index refusals(at::Device device,at::ScalarType dtype) {
  auto f=component_fixture(3);test::model_dtype(f.model,dtype);ContentProfile profile(f.graph,f.model,device);Index cases=0;
  for(int kind=0;kind<3;++kind) {
    bool refused=false;try{PackedSwiGluFull full(profile,device,kind==0?0:8,kind==1?0:4,kind==2?1:1024*1024);}
    catch(const std::invalid_argument&){refused=true;}
    require(refused,"SwiGLU accepted invalid dimensions/budget");++cases;
  }
  PackedSwiGluFull full(profile,device,1,1,1024*1024);auto longs=at::TensorOptions().device(device).dtype(at::kLong);
  ActionBatch input{at::zeros({1,4},longs),at::zeros({1,3},longs.dtype(dtype)),at::ones({1},longs.dtype(at::kBool))};
  input.coordinates[0][1].fill_(99);auto error=at::zeros({1},longs.dtype(at::kInt)),chunks=at::zeros({1},longs);
  CannProgram p(device);full.append_stage(p,input,input.values,input.values,error,chunks);p.finish();p.run();
  require(error.cpu().item<int>()==2,"SwiGLU invalid selected owner did not refuse");return cases+1;
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);
    if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||(args.dtype!=at::kFloat&&args.dtype!=at::kHalf))throw std::invalid_argument("SwiGLU gate requires explicit NPU FP32/FP16");
    args.allow_npu_float16=true;
    auto device=portable_torch::resolve_device(args);if(device.type()!=c10::DeviceType::PrivateUse1)throw std::invalid_argument("SwiGLU gate requires NPU");
    at::set_num_threads(1);at::set_num_interop_threads(1);at::NoGradGuard guard;
    const auto a=components(device,args.dtype),b=args.dtype==at::kFloat?windows(device):0,c=refusals(device,args.dtype);
    std::cout<<"device-swiglu: passed components="<<a<<" windows="<<b<<" refusals="<<c<<" scope="<<(args.dtype==at::kFloat?"FP32_HARD_inference":"FP16_component_only")<<" CPU=storage_dtype_FP64 keep_dtype=true\n";
    runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
