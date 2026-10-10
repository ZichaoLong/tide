#include "device_backend.h"
#include "retained_fixture.h"
#include "full_vjp_fixture.h"
#include "precision_graph_fixture.h"
#include "precision_graph_profiles.h"
#include "retained_cache_fixture.h"
#include "parameter_vjp.h"
#include "portable_torch/runtime.hpp"
#include "../../cpp/bench/streaming.h"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <iostream>
#include <limits>

namespace {
using namespace tide;using namespace tide::device_online;
void require(bool x,const char* message){if(!x)throw std::runtime_error(message);}
void trajectory(at::Device device,int shape,int variant,int64_t width,bool prefill,int mode,at::ScalarType payload,std::string emit="hard",int profile=0,int cache=-1,bool clock=false) {
  at::NoGradGuard guard;auto f=cache<0?test::precision_graph_profile(shape,variant,width,profile):
    test::retained_cache_fixture(shape,variant,width,cache,clock);
  if(emit!="hard") {
    for(auto& r:f.graph.regions)r.read_mode=shape==0?"old":"proposal";
    f.graph.compile();f.initial.identity=f.graph.identity;
  }
  test::fixture_dtype(f,payload);auto registry=f.model.parameters(false);
  const bool half=payload==at::kHalf;
  ContentLimits limits;limits.prefill=prefill;limits.trace=512;limits.full_chunk_rows=3;if(width>3)limits.workspace_bytes=512*1024*1024;
  if(cache>=0){limits.queue=96;limits.arrivals=192;limits.outputs=192;limits.kv_rows=64;limits.kv_trace_rows=8192;
    limits.attention_key_rows=2;limits.attention_chunk_rows=3;}
  limits.mode=emit;
  ContentFlow flow(f.graph,f.model,f.initial,device,limits);std::vector<RetainedTape> saved;std::vector<Result> forward;
  int64_t cut=f.initial.cut;ReverseTape live{};
  for(auto stop:test::retained_stops(cut)) {
    std::vector<External> input;for(const auto& x:f.input)if(x.time>=cut&&x.time<stop)input.push_back(x);
    flow.advance_device(input,stop);live=flow.reverse_tape();saved.push_back(retain_reverse_tape(live,(cache>=0&&width>64?256:64)*1024*1024));forward.push_back(flow.result());cut=stop;
    require(saved.back().tensor_bytes==reverse_tape_bytes(live),"retained byte accounting differs from actual tensors");
    require(saved.back().tape.source_scales.scalar_type()==payload&&saved.back().tape.state.values.scalar_type()==at::kFloat,
      "retained tape changed forward/journal precision");
  }
  flow.close();
  // Saved records remain valid after close and after all live journal/parameter
  // buffers have been overwritten. Only numerical tensors are touched here.
  live.state.values.fill_(std::numeric_limits<float>::quiet_NaN());live.state.count.fill_(-1);
  if(live.full.has_tanh)live.full.weights.fill_(std::numeric_limits<float>::quiet_NaN());
  test::poison_live_cache(live);
  auto error=at::zeros({1},saved[0].tape.state.count.options().dtype(at::kInt));DeviceProgram p(device);p.limit_workspace(128*1024*1024);
  // Wide SwiGLU needs at least one matrix row plus its parameter accumulators
  // within the nested Full reserve. This is an explicit admission budget.
  const Index reverse_bytes=cache>=0?Index(width>64?8:1)*1024*1024*1024:(width>3?256:128)*1024*1024;
  std::vector<GraphCotangents> roots;for(size_t i=0;i<saved.size();++i) {
    auto r=test::retained_roots(saved[i].tape,i,mode);test::retained_cache_roots(r,saved[i].tape,i,mode);roots.push_back(r);
  }
  std::vector<GraphVjp> gradients(saved.size());ParameterVjp total;
  for(size_t i=saved.size();i>0;) {--i;auto cot=roots[i];
    if(i+1<saved.size())cot=append_window_bridge(p,saved[i].tape,cot,saved[i+1].tape,gradients[i+1],error,16*1024*1024);
    gradients[i]=append_graph_vjp(p,saved[i].tape,cot,error,3,reverse_bytes);
    auto partial=append_parameter_vjp(p,f.graph,registry,gradients[i],error,16*1024*1024);
    total=total.values.defined()?append_parameter_accumulate(p,total,partial,error,16*1024*1024):partial;
  }
  p.finish();portable_torch::synchronize(device);p.run();require(!error.cpu().item<int>(),"valid retained graph reverse refused");
  auto values=total.values.cpu(),on=total.connected.cpu();
  auto cache_values=test::cache_gradient_values(gradients.front());
  for(auto dtype:{at::kFloat,at::kDouble}) {
    Options options;options.mode=emit;auto expected=test::retained_reference_precision(f,mode,dtype,half,options);
    if(dtype==at::kFloat)for(size_t i=0;i<saved.size();++i)tide_bench::compare(forward[i],expected.windows[i],true,payload,
      std::nullopt,half?2e-2:1e-5,half?2e-3:1e-6);
    for(size_t i=0;i<total.owners.size();++i) {
      const auto& owner=total.owners[i];auto value=total.offsets[i]<0?at::zeros(owner.value.sizes(),at::kFloat):
        values.narrow(0,total.offsets[i],owner.value.numel()).reshape(owner.value.sizes());
      test::full_same_precision(value,on[i],expected.gradients.at(owner.canonical),owner.canonical.c_str(),half);
    }
    auto initial=gradients.front().initial.cpu(),ic=gradients.front().initial_connected.cpu();
    for(const auto& [owner,_]:f.initial.states) {
      const auto name="state/"+std::to_string(owner.first)+"/"+std::to_string(owner.second);
      test::full_same_precision(initial[owner.first][owner.second],ic[owner.first][owner.second],expected.gradients.at(name),name.c_str(),half);
    }
    test::compare_retained_cache(gradients.front(),saved.front().tape,expected,f,half);
    std::map<std::string,bool> seen;
    for(size_t w=0;w<saved.size();++w) {
      const auto& g=gradients[w];auto meta=g.links.messages.cpu(),valid=g.links.valid.cpu(),value=g.messages.cpu(),connected=g.message_connected.cpu();
      auto fm=saved[w].tape.fiber_meta.cpu(),pm=saved[w].tape.pending.coordinates.cpu();
      for(Index i=0;i<g.links.fibers+g.links.pending;++i)if(valid[i].item<bool>()&&meta[i][1].item<Index>()<0) {
        const auto row=i<g.links.fibers?fm[i]:pm[i-g.links.fibers];if(row[3].item<Index>()!=0)continue;
        Atom a{row[0].item<Index>(),row[1].item<Index>(),row[2].item<Index>(),0,row[4].item<Index>(),row[5].item<Index>(),{}};
        const auto name=test::boundary_name(a);require(!seen[name],"external leaf appeared in two windows");seen[name]=true;
        test::full_same_precision(value[i],connected[i],expected.gradients.at(name),name.c_str(),half);
      }
    }
    for(const auto& [name,_]:expected.gradients)if(name.rfind("boundary/",0)==0)require(seen[name],"retained reverse lost an external leaf");
  }
  portable_torch::synchronize(device);p.run();require(!error.cpu().item<int>()&&at::equal(total.values.cpu(),values)&&at::equal(total.connected.cpu(),on),"retained reverse replay changed gradients");
  auto replay_cache=test::cache_gradient_values(gradients.front());require(replay_cache.size()==cache_values.size(),"cache replay changed groups");
  for(size_t i=0;i<cache_values.size();++i)require(at::equal(cache_values[i],replay_cache[i]),"retained cache replay changed gradients");
  if(cache<0&&!profile&&shape==0&&variant==0&&width==3&&!prefill&&mode==4) {
    auto reject=[&](auto fn){bool failed=false;try{fn();}catch(const std::invalid_argument&){failed=true;}require(failed,"retained preflight refusal missing");};
    reject([&]{retain_reverse_tape(saved[0].tape,1);});
    reject([&]{DeviceProgram small(device);append_window_bridge(small,saved[0].tape,roots[0],saved[1].tape,gradients[1],error,1);});
    auto wrong=saved[1].tape;++wrong.cut;
    reject([&]{DeviceProgram bad(device);append_window_bridge(bad,saved[0].tape,roots[0],wrong,gradients[1],error,16*1024*1024);});
    reject([&]{DeviceProgram small(device);append_parameter_accumulate(small,total,total,error,1);});
    const auto flags=saved[0].tape.pending.valid.cpu();Index changed=-1;
    for(Index i=0;i<flags.numel();++i)if(flags[i].item<bool>()){changed=i;break;}
    require(changed>=0,"fixture has no cross-window pending message");
    auto old=saved[0].tape.pending.coordinates[changed].clone();saved[0].tape.pending.coordinates[changed][2].add_(12345);
    DeviceProgram missing(device);append_window_bridge(missing,saved[0].tape,roots[0],saved[1].tape,gradients[1],error,16*1024*1024);missing.finish();
    portable_torch::synchronize(device);missing.run();require(error.cpu().item<int>()==22,"missing boundary message was silently dropped");
    saved[0].tape.pending.coordinates[changed].copy_(old);
  }
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    bool extended=false,event=false,fiber=false,smoke=false;std::vector<char*> forwarded{argv[0]};
    for(int i=1;i<argc;++i) {
      const std::string arg=argv[i];
      if(arg=="--extended")extended=true;else if(arg=="--event-cache")event=true;else if(arg=="--fiber-cache")fiber=true;
      else if(arg=="--profile-smoke")smoke=true;else forwarded.push_back(argv[i]);
    }
    if(int(extended)+int(event)+int(fiber)>1||(smoke&&!event&&!fiber))throw std::invalid_argument("incompatible retained checker profiles");
    auto args=portable_torch::parse_cli(forwarded.size(),forwarded.data(),true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||(args.dtype!=at::kFloat&&args.dtype!=at::kHalf))throw std::invalid_argument("retained gate requires explicit NPU FP32/FP16");
    args.allow_npu_float16=true;
    const auto device=portable_torch::resolve_device(args);if(device.type()!=tide::device_online::resident_device_type)throw std::invalid_argument("retained gate requires NPU");
    at::set_num_threads(1);at::set_num_interop_threads(1);int cases=0;
    if(event||fiber) {
      auto run=[&](int shape,int variant,Index width,bool prefill,int mode,int cache,const std::string& emit="hard",bool clock=false) {
        try{trajectory(device,shape,variant,width,prefill,mode,args.dtype,emit,0,cache,clock);++cases;}
        catch(...){std::cerr<<"cache retained profile="<<cache<<" shape="<<shape<<" variant="<<variant<<" width="<<width
          <<" prefill="<<prefill<<" roots="<<mode<<" emit="<<emit<<" clock="<<clock<<'\n';throw;}
      };
      if(smoke) {
        run(0,1,4,true,9,event?0:6,"hst");
        if(fiber)run(2,0,7,true,9,5,"softp",true);
      } else if(event) {
        for(int shape:{0,3})for(int variant:{0,1})for(bool prefill:{false,true})for(int mode:{0,1,4,6,7,8,9})run(shape,variant,4,prefill,mode,0);
        for(const std::string emit:{"hst","softp"})for(int variant:{0,1})for(bool prefill:{false,true})run(0,variant,4,prefill,9,0,emit);
        for(Index width:{1,257})run(3,0,width,true,9,0);
      } else {
        for(int cache=1;cache<=5;++cache)for(int variant:{0,1})for(bool prefill:{false,true})for(int mode:{0,1,6,7,8,9,10})run((cache+variant)%2?0:3,variant,4,prefill,mode,cache);
        for(const std::string emit:{"hst","softp"})for(int variant:{0,1})for(bool prefill:{false,true})run(0,variant,4,prefill,9,5,emit);
        for(bool prefill:{false,true})run(0,1,4,prefill,9,6);
        for(Index width:{1,257})run(2,0,width,true,9,5,"hard",true);
      }
      std::cout<<"device-"<<(event?"event":"fiber")<<"-retained: passed trajectories="<<cases<<" windows="<<cases*4
        <<" dtype="<<args.dtype<<" CPU=FP32_FP64 after_close=true replay=true scope="<<(smoke?"profile-smoke":"retained_cache_backward")<<'\n';
      runtime.close();return 0;
    }
    if(extended) {
      test::check_half_reference_norm();
      for(int profile=1;profile<=16;++profile)for(bool prefill:{false,true})for(int mode:{0,4,5}) {
        try{trajectory(device,profile%2?0:3,0,3,prefill,mode,args.dtype,"hard",profile);++cases;}
        catch(...){std::cerr<<"extended retained profile="<<profile<<" prefill="<<prefill<<" roots="<<mode<<'\n';throw;}
      }
      for(int profile:{4,11,16})for(const std::string emit:{"hst","softp"})for(bool prefill:{false,true}) {
        try{trajectory(device,0,1,3,prefill,4,args.dtype,emit,profile);++cases;}
        catch(...){std::cerr<<"extended retained profile="<<profile<<" prefill="<<prefill<<" emit="<<emit<<'\n';throw;}
      }
      for(int profile:{4,15}) {
        try{trajectory(device,0,0,257,true,4,args.dtype,"hard",profile);++cases;}
        catch(...){std::cerr<<"extended retained profile="<<profile<<" width=257\n";throw;}
      }
      std::cout<<"device-extended-retained: passed trajectories="<<cases<<" windows="<<cases*4<<" dtype="<<args.dtype
        <<" CPU=FP32_FP64 after_close=true replay=true scope=normalized_Aggregate_LH_SwiGLU_retained_backward\n";
      runtime.close();return 0;
    }
    for(int shape:{0,3})for(bool prefill:{false,true})for(int mode=0;mode<6;++mode) {
      try{trajectory(device,shape,0,3,prefill,mode,args.dtype);++cases;}
      catch(...){std::cerr<<"retained shape="<<shape<<" prefill="<<prefill<<" mode="<<mode<<'\n';throw;}
    }
    trajectory(device,0,0,257,true,4,args.dtype);++cases;trajectory(device,3,1,3,true,4,args.dtype);++cases;
    for(const std::string emit:{"hst","softp"})for(int shape:{0,3})for(bool prefill:{false,true})for(int mode:{4,5}) {
      try{trajectory(device,shape,0,3,prefill,mode,args.dtype,emit);++cases;}
      catch(...){std::cerr<<"retained control shape="<<shape<<" prefill="<<prefill<<" roots="<<mode<<" mode="<<emit<<'\n';throw;}
    }
    std::cout<<"device-retained: passed trajectories="<<cases<<" windows="<<cases*4<<" dtype="<<args.dtype<<" CPU=FP32_FP64 after_close=true replay=true scope=internal_retained_HARD_HST_SOFTP_backward\n";
    runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
