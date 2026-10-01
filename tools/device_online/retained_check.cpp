#include "retained_fixture.h"
#include "full_vjp_fixture.h"
#include "precision_graph_fixture.h"
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
void trajectory(at::Device device,int shape,int variant,int64_t width,bool prefill,int mode,at::ScalarType payload,std::string emit="hard") {
  at::NoGradGuard guard;auto f=test::retained_fixture(shape,variant,width);
  if(emit!="hard") {
    for(auto& r:f.graph.regions)r.read_mode=shape==0?"old":"proposal";
    f.graph.compile();f.initial.identity=f.graph.identity;
  }
  test::fixture_dtype(f,payload);auto registry=f.model.parameters(false);
  const bool half=payload==at::kHalf;
  ContentLimits limits;limits.prefill=prefill;limits.trace=512;limits.full_chunk_rows=3;if(width>3)limits.workspace_bytes=512*1024*1024;
  limits.mode=emit;
  ContentFlow flow(f.graph,f.model,f.initial,device,limits);std::vector<RetainedTape> saved;std::vector<Result> forward;
  int64_t cut=f.initial.cut;ReverseTape live{};
  for(auto stop:test::retained_stops(cut)) {
    std::vector<External> input;for(const auto& x:f.input)if(x.time>=cut&&x.time<stop)input.push_back(x);
    flow.advance_device(input,stop);live=flow.reverse_tape();saved.push_back(retain_reverse_tape(live,64*1024*1024));forward.push_back(flow.result());cut=stop;
    require(saved.back().tensor_bytes==reverse_tape_bytes(live),"retained byte accounting differs from actual tensors");
    require(saved.back().tape.source_scales.scalar_type()==payload&&saved.back().tape.state.values.scalar_type()==at::kFloat,
      "retained tape changed forward/journal precision");
  }
  flow.close();
  // Saved records remain valid after close and after all live journal/parameter
  // buffers have been overwritten. Only numerical tensors are touched here.
  live.state.values.fill_(std::numeric_limits<float>::quiet_NaN());live.state.count.fill_(-1);
  if(live.full.has_tanh)live.full.weights.fill_(std::numeric_limits<float>::quiet_NaN());
  auto error=at::zeros({1},saved[0].tape.state.count.options().dtype(at::kInt));CannProgram p(device);p.limit_workspace(128*1024*1024);
  std::vector<GraphCotangents> roots;for(size_t i=0;i<saved.size();++i)roots.push_back(test::retained_roots(saved[i].tape,i,mode));
  std::vector<GraphVjp> gradients(saved.size());ParameterVjp total;
  for(size_t i=saved.size();i>0;) {--i;auto cot=roots[i];
    if(i+1<saved.size())cot=append_window_bridge(p,saved[i].tape,cot,saved[i+1].tape,gradients[i+1],error,16*1024*1024);
    gradients[i]=append_graph_vjp(p,saved[i].tape,cot,error,3,128*1024*1024);
    auto partial=append_parameter_vjp(p,f.graph,registry,gradients[i],error,16*1024*1024);
    total=total.values.defined()?append_parameter_accumulate(p,total,partial,error,16*1024*1024):partial;
  }
  p.finish();portable_torch::synchronize(device);p.run();require(!error.cpu().item<int>(),"valid retained graph reverse refused");
  auto values=total.values.cpu(),on=total.connected.cpu();
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
  if(shape==0&&variant==0&&width==3&&!prefill&&mode==4) {
    auto reject=[&](auto fn){bool failed=false;try{fn();}catch(const std::invalid_argument&){failed=true;}require(failed,"retained preflight refusal missing");};
    reject([&]{retain_reverse_tape(saved[0].tape,1);});
    reject([&]{CannProgram small(device);append_window_bridge(small,saved[0].tape,roots[0],saved[1].tape,gradients[1],error,1);});
    auto wrong=saved[1].tape;++wrong.cut;
    reject([&]{CannProgram bad(device);append_window_bridge(bad,saved[0].tape,roots[0],wrong,gradients[1],error,16*1024*1024);});
    reject([&]{CannProgram small(device);append_parameter_accumulate(small,total,total,error,1);});
    const auto flags=saved[0].tape.pending.valid.cpu();Index changed=-1;
    for(Index i=0;i<flags.numel();++i)if(flags[i].item<bool>()){changed=i;break;}
    require(changed>=0,"fixture has no cross-window pending message");
    auto old=saved[0].tape.pending.coordinates[changed].clone();saved[0].tape.pending.coordinates[changed][2].add_(12345);
    CannProgram missing(device);append_window_bridge(missing,saved[0].tape,roots[0],saved[1].tape,gradients[1],error,16*1024*1024);missing.finish();
    portable_torch::synchronize(device);missing.run();require(error.cpu().item<int>()==22,"missing boundary message was silently dropped");
    saved[0].tape.pending.coordinates[changed].copy_(old);
  }
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||(args.dtype!=at::kFloat&&args.dtype!=at::kHalf))throw std::invalid_argument("retained gate requires explicit NPU FP32/FP16");
    args.allow_npu_float16=true;
    const auto device=portable_torch::resolve_device(args);if(device.type()!=c10::DeviceType::PrivateUse1)throw std::invalid_argument("retained gate requires NPU");
    at::set_num_threads(1);at::set_num_interop_threads(1);int cases=0;
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
