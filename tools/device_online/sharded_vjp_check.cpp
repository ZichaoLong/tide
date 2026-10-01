#include "sharded_vjp_compare.h"
#include "precision_graph_profiles.h"
#include "precision_graph_fixture.h"
#include "retained_cache_fixture.h"
#include "sharded_vjp_boundaries.h"
#include "portable_torch/runtime.hpp"
#include "../../cpp/bench/streaming.h"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <iostream>
#include <limits>
namespace {
using namespace tide;using namespace tide::device_online;
void require(bool x,const char* message){if(!x)throw std::runtime_error(message);}
void check(at::Device device,int devices,at::ScalarType dtype,int shape,int variant,int profile,int cache,
           bool prefill,int mode,const std::string& emit,const std::string& policy,Index width=4) {
  at::NoGradGuard guard;
  auto f=cache<0?test::precision_graph_profile(shape,variant,width,profile):test::retained_cache_fixture(shape,variant,width,cache);
  if(emit!="hard"){for(auto& r:f.graph.regions)r.read_mode=shape==0?"old":"proposal";f.graph.compile();f.initial.identity=f.graph.identity;}
  test::fixture_dtype(f,dtype);auto registry=f.model.parameters(false);const bool half=dtype==at::kHalf;
  ContentLimits limits;limits.prefill=prefill;limits.mode=emit;limits.trace=512;limits.full_chunk_rows=3;
  if(cache>=0){limits.queue=96;limits.arrivals=192;limits.outputs=192;limits.kv_rows=64;limits.kv_trace_rows=8192;limits.attention_key_rows=2;limits.attention_chunk_rows=3;}
  if(width>64)limits.workspace_bytes=1024*1024*1024;
  std::vector<at::Device> placement;for(int i=0;i<devices;++i)placement.emplace_back(device.type(),device.index()+i);
  ContentFlow flow(f.graph,f.model,f.initial,device,limits,place_full(f.graph,f.model,placement,policy));
  std::vector<RetainedShardedTape> saved;std::vector<Result> forward;ShardedReverseTape live{};
  auto cut=f.initial.cut;
  for(auto stop:test::retained_stops(cut)) {
    std::vector<External> input;for(const auto& x:f.input)if(x.time>=cut&&x.time<stop)input.push_back(x);
    flow.advance_device(input,stop);live=flow.sharded_reverse_tape();saved.push_back(retain_sharded_reverse_tape(live,512*1024*1024));
    forward.push_back(flow.result());cut=stop;
    require(saved.back().tensor_bytes==sharded_reverse_tape_bytes(live),"sharded tape byte accounting mismatch");
  }
  flow.close();live.coordinator.state.values.fill_(std::numeric_limits<float>::quiet_NaN());live.coordinator.state.count.fill_(-1);
  for(auto& shard:live.shards)for(auto& x:{shard.full.weights,shard.full.biases,shard.full.extra.lh_weights,shard.full.extra.gate,shard.full.extra.up,shard.full.extra.down})
    if(x.defined())x.fill_(std::numeric_limits<float>::quiet_NaN());
  test::poison_live_cache(live.coordinator);
  auto error=at::zeros({1},saved[0].tape.coordinator.state.count.options().dtype(at::kInt));
  CannProgram p(device);p.limit_workspace(128*1024*1024);std::vector<ShardedGraphVjp> gradients(saved.size());
  for(size_t i=saved.size();i>0;) {--i;const auto& t=saved[i].tape.coordinator;
    auto roots=test::retained_roots(t,i,mode);test::retained_cache_roots(roots,t,i,mode);
    if(i+1<saved.size())roots=append_window_bridge(p,t,roots,saved[i+1].tape.coordinator,gradients[i+1].coordinator,error,32*1024*1024);
    gradients[i]=append_sharded_graph_vjp(p,saved[i].tape,roots,error,prefill?3:1,Index(width>64?4:1)*1024*1024*1024,128*1024*1024);
  }
  p.finish();run_sharded_graph_vjp(p,gradients);require(!error.cpu().item<int>(),"sharded graph reverse refused");
  const auto values=test::sharded_owner_observations(f.graph,registry,gradients);
  for(auto reference_dtype:{at::kFloat,at::kDouble}) {
    Options options;options.mode=emit;auto expected=test::retained_reference_precision(f,mode,reference_dtype,half,options);
    for(const auto& [name,value]:values)test::full_same_precision(value.defined()?value:at::zeros({},at::kFloat),at::full({},value.defined(),at::kBool),expected.gradients.at(name),name.c_str(),half);
    if(reference_dtype==at::kFloat)for(size_t i=0;i<saved.size();++i)tide_bench::compare(forward[i],expected.windows[i],true,dtype,std::nullopt,half?2e-2:1e-5,half?2e-3:1e-6);
    const auto& first=gradients.front().coordinator;auto initial=first.initial.cpu(),on=first.initial_connected.cpu();
    for(const auto& [owner,_]:f.initial.states) {
      const auto name="state/"+std::to_string(owner.first)+"/"+std::to_string(owner.second);
      test::full_same_precision(initial[owner.first][owner.second],on[owner.first][owner.second],expected.gradients.at(name),name.c_str(),half);
    }
    test::compare_retained_cache(first,saved.front().tape.coordinator,expected,f,half);
    std::map<std::string,bool> seen;
    for(size_t w=0;w<saved.size();++w) {
      const auto& g=gradients[w].coordinator;const auto& t=saved[w].tape.coordinator;
      auto meta=g.links.messages.cpu(),valid=g.links.valid.cpu(),v=g.messages.cpu(),c=g.message_connected.cpu();
      auto fm=t.fiber_meta.cpu(),pm=t.pending.coordinates.cpu();
      require(g.reverse_stages.cpu().item<Index>()==g.links.stages.cpu().item<Index>(),"incomplete sharded reverse stage progression");
      for(Index i=0;i<g.links.fibers+g.links.pending;++i)if(valid[i].item<bool>()&&meta[i][1].item<Index>()<0) {
        auto row=i<g.links.fibers?fm[i]:pm[i-g.links.fibers];if(row[3].item<Index>()!=0)continue;
        Atom atom{row[0].item<Index>(),row[1].item<Index>(),row[2].item<Index>(),0,row[4].item<Index>(),row[5].item<Index>(),{}};
        const auto name=test::boundary_name(atom);require(!seen[name],"duplicate external boundary gradient");seen[name]=true;
        test::full_same_precision(v[i],c[i],expected.gradients.at(name),name.c_str(),half);
      }
    }
    for(const auto& [name,_]:expected.gradients)if(name.rfind("boundary/",0)==0)require(seen[name],"missing external boundary gradient");
  }
  run_sharded_graph_vjp(p,gradients);require(!error.cpu().item<int>(),"sharded graph replay refused");
  auto replay=test::sharded_owner_observations(f.graph,registry,gradients);
  for(const auto& [name,value]:values)require(value.defined()==replay.at(name).defined()&&(!value.defined()||at::equal(value,replay.at(name))),"sharded reverse replay accumulated stale gradients");
  p.close();for(auto& g:gradients)g.full->close();
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    int devices=2;bool smoke=false;std::string policy="locality";std::vector<char*> forwarded{argv[0]};
    for(int i=1;i<argc;++i){std::string arg=argv[i];if(arg.rfind("--full-shards=",0)==0)devices=std::stoi(arg.substr(14));
      else if(arg=="--profile-smoke")smoke=true;else if(arg.rfind("--full-placement=",0)==0)policy=arg.substr(17);else forwarded.push_back(argv[i]);}
    auto args=portable_torch::parse_cli(forwarded.size(),forwarded.data(),true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||(args.dtype!=at::kFloat&&args.dtype!=at::kHalf)||devices<1||devices>4)throw std::invalid_argument("sharded VJP gate requires explicit NPU FP32/FP16 and 1..4 Full owners");
    args.allow_npu_float16=true;const auto d=portable_torch::resolve_device(args);if(d.type()!=c10::DeviceType::PrivateUse1)throw std::invalid_argument("sharded VJP gate requires NPU");
    at::set_num_threads(1);at::set_num_interop_threads(1);int cases=0;
    auto run=[&](int shape,int variant,int profile,int cache,bool prefill,int mode,const std::string& emit,Index width=4) {
      try{check(d,devices,args.dtype,shape,variant,profile,cache,prefill,mode,emit,policy,width);++cases;}
      catch(...){std::cerr<<"sharded reverse shape="<<shape<<" variant="<<variant<<" profile="<<profile<<" cache="<<cache<<" prefill="<<prefill<<" roots="<<mode<<" emit="<<emit<<'\n';throw;}
    };
    if(smoke){run(0,1,16,-1,true,4,"hst");run(0,1,0,6,true,9,"softp");}
    else {
      for(int shape=0;shape<4;++shape)for(bool prefill:{false,true})for(int mode:{0,4,5})run(shape,shape%2,0,-1,prefill,mode,"hard");
      for(int profile=1;profile<=16;++profile)run(profile%2,1,profile,-1,profile%2,4,profile%2?"hst":"softp");
      for(int cache:{0,1,5,6})for(bool prefill:{false,true})run(0,1,0,cache,prefill,9,prefill?"hst":"softp");
      for(Index width:{1,257})run(0,1,16,-1,true,4,"hard",width);
      test::sharded_vjp_boundaries(d,devices,args.dtype);
    }
    std::cout<<"sharded-graph-vjp: passed trajectories="<<cases<<" windows="<<cases*4<<" devices="<<devices
      <<" dtype="<<args.dtype<<" CPU=FP32_FP64 after_close=true replay=true scope="<<(smoke?"profile-smoke":"Full_owner_partials_retained_graph_not_optimizer")<<'\n';
    runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
