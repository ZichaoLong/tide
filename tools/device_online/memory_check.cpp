#include "content_fixture.h"
#include "content_profile.h"
#include "content_budget.h"
#include "portable_torch/runtime.hpp"
#include "tide/stream.h"
#include "../../cpp/bench/streaming.h"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using namespace tide;
using namespace tide::device_online;
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
template<class F> void refuses(F f,const std::string& marker) {
  try{f();}catch(const std::exception& e){if(std::string(e.what()).find(marker)!=std::string::npos)return;throw;}
  throw std::runtime_error("missing memory refusal: "+marker);
}
test::Fixture fixture(Index width) {
  test::Fixture f;auto& g=f.graph;
  for(Index i=0;i<6;++i) {
    g.nodes.push_back({i});g.regions.push_back({1,true,false,"proposal","count-v1"});
    g.inputs.push_back(i);g.outputs.push_back(i);g.edges.push_back({i,(i+1)%6,1+i%3});
  }
  g.nodes[0].memory=g.nodes[5].memory="attention";g.nodes[0].window=3;
  g.nodes[1].memory="lh-fiber-attention-mean-repeat-v1";
  g.nodes[2].memory="identity";g.nodes[3].memory="lh-add-repeat-v1";g.nodes[4].memory="ema";
  g.nodes[0].full=g.nodes[5].full="swiglu";g.nodes[1].full="lh-silu-layer-v1";
  g.nodes[2].full="tanh";g.nodes[3].full="lh-relu-identity-v1";g.nodes[4].full="identity";
  for(auto& n:g.nodes){n.emission="slot_affine";n.query_heads=width==33?3:1;n.kv_heads=n.memory=="attention"?1:n.query_heads;}
  g.compile();const auto eye=at::eye(width,at::kFloat);
  for(Index n=0;n<6;++n) {
    NodeWeights w{at::zeros({width},at::kFloat),eye*.125f,at::full({width},.015625f,at::kFloat),at::ones({width},at::kFloat)/width};
    if(g.nodes[n].memory=="attention") {
      const auto kv=width/g.nodes[n].query_heads;
      w.extra["attn_q"]=eye*.25f;w.extra["attn_k"]=eye.narrow(1,0,kv)*.5f;
      w.extra["attn_v"]=eye.narrow(1,0,kv).clone();w.extra["attn_out"]=eye*.5f;
    }
    if(n==1) {
      w.extra["fiber_qkv"]=at::cat({eye*.25f,eye*.5f,eye},1);w.extra["fiber_qkv_bias"]=at::zeros({3*width},at::kFloat);
      w.extra["fiber_out"]=eye*.5f;w.extra["fiber_out_bias"]=at::zeros({width},at::kFloat);
      w.extra["fiber_decay"]=at::full({},.01f,at::kFloat);
      w.extra["lh_norm_weight"]=at::ones({width},at::kFloat);w.extra["lh_norm_bias"]=at::zeros({width},at::kFloat);
    }
    if(n==3)w.extra["add_retention"]=at::full({},.75f,at::kFloat);
    if(n==0||n==5) {
      w.extra["ffn_gate"]=at::cat({eye*.1f,eye*.2f},1);w.extra["ffn_up"]=at::cat({eye*.25f,eye*.125f},1);
      w.extra["ffn_down"]=at::cat({eye*.5f,eye*.25f},0);
    }
    for(Index slot=0;slot<g.outgoing_ports.offsets[n+1]-g.outgoing_ports.offsets[n];++slot) {
      w.extra["emit_w_"+std::to_string(slot)]=eye*.25f;w.extra["emit_b_"+std::to_string(slot)]=at::zeros({width},at::kFloat);
    }
    f.model.nodes.push_back(w);f.model.input_scale.push_back(at::ones({},at::kFloat));
    f.model.output_scale.push_back(at::ones({},at::kFloat));f.model.agg_scale.push_back(at::ones({},at::kFloat));
    f.model.edge_scale.push_back(at::ones({},at::kFloat));
  }
  f.initial.identity=g.identity;f.initial.batch_size=2;
  for(Index b=0;b<2;++b)for(Index n=0;n<6;++n) {
    Index position=0;for(Index t:{0,2,5})
      // Nonuniform channels avoid a nearly cancelled middle LayerNorm output
      // in this allocation fixture. The separate LH gate tests conditioning.
      f.input.push_back({b,n,position++,t,at::arange(width,at::kFloat).remainder(7).square()*.0625f+.125f+float(b+n+t)*.015625f});
  }
  return f;
}
ContentLimits limits(Index bytes,ChunkPolicy policy) {
  ContentLimits l;l.queue=128;l.arrivals=256;l.outputs=256;l.trace=512;
  l.kv_rows=64;l.kv_trace_rows=512;l.attention_key_rows=7;l.attention_chunk_rows=32;
  l.full_chunk_rows=64;l.emission_chunk_rows=128;l.workspace_bytes=bytes;l.chunk_policy=policy;return l;
}
Index windows(at::Device device) {
  Index cases=0;
  for(Index width:{3,33,257})for(Index budget:{192*1024*1024,512*1024*1024})
  for(auto policy:{ChunkPolicy::conservative,ChunkPolicy::aggressive})for(bool prefill:{false,true}) {
    auto f=fixture(width);auto l=limits(budget,policy);l.prefill=prefill;
    portable_torch::synchronize(device);
    const auto baseline=c10_npu::NPUCachingAllocator::getDeviceStats(device.index());
    c10_npu::NPUCachingAllocator::resetPeakStats(device.index());
    ContentFlow flow(f.graph,f.model,f.initial,device,l);Streaming cpu(f.graph,f.model,{});
    auto q=f.initial;Index previous=0;std::map<std::string,Index> observed;
    for(Index stop:{3,7,11}) {
      std::vector<External> xs;for(const auto& x:f.input)if(x.time>=previous&&x.time<stop)xs.push_back(x);
      auto expected=cpu.run(q,xs,stop,stop),actual=flow.advance(xs,stop);
      try{tide_bench::compare(actual,expected,true,at::kFloat);}
      catch(const std::exception&){
        std::cerr<<"memory width="<<width<<" budget="<<budget<<" aggressive="<<(policy==ChunkPolicy::aggressive)<<" stop="<<stop<<'\n';
        if(actual.trace.size()==expected.trace.size())for(size_t i=0;i<actual.trace.size();++i) {
          const auto& a=actual.trace[i];const auto& b=expected.trace[i];
          if(a.full.defined()&&b.full.defined()&&!at::allclose(a.full,b.full,1e-5,1e-6)) {
            std::cerr<<"full mismatch node="<<a.node<<" time="<<a.time<<" input="<<a.comparison<<" reference_input="<<b.comparison
              <<" actual="<<a.full<<" expected="<<b.full<<'\n';
            if(f.graph.nodes[a.node].full=="lh-silu-layer-v1") {
              auto x=at::silu(b.comparison.to(at::kDouble));auto centered=x-x.mean();
              auto exact=centered/at::sqrt(centered.square().mean()+1e-5);
              std::cerr<<"FP64 from reference input="<<exact<<" variance="<<centered.square().mean().item<double>()<<'\n';
            }
          }
        }
        throw;
      }
      const auto& s=actual.stats;
      require(s.at("retained_tensor_bytes")<=s.at("planned_buffer_bytes"),"observed tensor storage exceeds estimate");
      require(s.at("planned_buffer_bytes")+s.at("cann_workspace_bytes")<=s.at("usable_memory_budget_bytes"),"CANN workspace escaped shared budget");
      require(s.at("usable_memory_budget_bytes")<budget&&s.at("planned_headroom_bytes")>0,"headroom absent");
      require(s.at("aggressive_chunking")==Index(policy==ChunkPolicy::aggressive),"effective chunk policy missing");
      observed=s;q=expected.continuation;previous=stop;++cases;
    }
    portable_torch::synchronize(device);
    const auto measured=c10_npu::NPUCachingAllocator::getDeviceStats(device.index());
    const auto peak=measured.allocated_bytes[0].peak-baseline.allocated_bytes[0].current;
    // Isolated fixture calibration, including construction transients. This is
    // TorchNPU allocator accounting, not all vendor/driver HBM or a free-HBM guarantee.
    require(peak<=budget,"measured allocator peak exceeds declared fixture budget");
    std::cout<<"memory-calibration {\"width\":"<<width<<",\"budget\":"<<budget
      <<",\"aggressive\":"<<Index(policy==ChunkPolicy::aggressive)<<",\"prefill\":"<<prefill
      <<",\"baseline_allocated\":"<<baseline.allocated_bytes[0].current
      <<",\"peak_allocated_delta\":"<<peak<<",\"peak_reserved\":"<<measured.reserved_bytes[0].peak;
    for(const auto* name:{"planned_buffer_bytes","retained_tensor_bytes","cann_workspace_bytes",
        "planned_headroom_bytes","emission_chunk_rows","full_chunk_rows","lh_full_chunk_rows",
        "swiglu_full_chunk_rows","attention_chunk_rows","attention_key_rows",
        "event_attention_chunk_rows","event_attention_key_rows"})
      std::cout<<",\""<<name<<"\":"<<observed.at(name);
    std::cout<<"}\n"<<std::flush;
  }
  return cases;
}
Index workspace(at::Device device) {
  const auto opts=at::TensorOptions().device(device).dtype(at::kFloat);
  auto value=at::zeros({8},opts),one=at::ones_like(value);Index needed=0;
  {
    CannProgram p(device);p.limit_workspace(0);
    refuses([&]{p.add(value,one);},"workspace budget exceeded before allocation");
    require(p.workspace_bytes()==0,"refused workspace was allocated");
    refuses([&]{p.finish();},"not open");p.close();
  }
  {
    CannProgram p(device);p.add(value,one);needed=p.workspace_bytes();
    // Two descriptors may view one storage. Count the actual allocation once.
    require(p.retained_tensor_bytes()==value.nbytes()+one.nbytes(),"tensor storage accounting duplicated aliases");
    p.add(value.narrow(0,0,4),one.narrow(0,0,4));
    require(p.retained_tensor_bytes()==value.nbytes()+one.nbytes(),"tensor views inflated allocated storage");
    p.finish();p.run();
    auto expected=at::ones({8},at::kFloat);expected.narrow(0,0,4).fill_(2.f);
    require(at::equal(value.cpu(),expected),"shared workspace corrupted sequential operations");
    p.close();require(p.workspace_bytes()==0&&p.retained_tensor_bytes()==0,"closed program retained allocation accounting");
  }
  value.zero_();portable_torch::synchronize(device);
  {
    CannProgram p(device);p.limit_workspace(needed);p.add(value,one);
    p.add(value,one);require(p.workspace_bytes()==needed,"serial workspace reserved a sum instead of a maximum");
    refuses([&]{p.limit_workspace(needed);},"before numerical operations");p.finish();p.run();
    require(at::equal(value.cpu(),at::full({8},2.f,at::kFloat)),"exact workspace limit changed computation");p.close();
  }
  auto f=fixture(3);ContentProfile profile(f.graph,f.model,device,true);
  require(profile.sources.device().is_cpu()&&profile.read.device().is_cpu(),"deferred validation uploaded device tables");
  auto l=limits(64*1024*1024,ChunkPolicy::aggressive);l.kv_rows=std::numeric_limits<Index>::max();
  refuses([&]{ContentFlow x(f.graph,f.model,f.initial,device,l);},"module minima exceed memory budget");
  l=limits(64*1024*1024,static_cast<ChunkPolicy>(99));
  refuses([&]{ContentFlow x(f.graph,f.model,f.initial,device,l);},"invalid limits");
  return 7;
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("memory gate requires explicit NPU FP32");
    auto device=portable_torch::resolve_device(args);if(device.type()!=c10::DeviceType::PrivateUse1)throw std::invalid_argument("memory gate requires NPU");
    at::set_num_threads(1);at::set_num_interop_threads(1);at::NoGradGuard guard;
    const auto a=workspace(device);std::cout<<"memory workspace/refusals="<<a<<" passed\n"<<std::flush;
    const auto b=windows(device);
    std::cout<<"device-memory: passed windows="<<b<<" workspace/refusals="<<a<<" scope=FP32_HARD_inference\n";
    runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
