#include "tide/resident.h"
#include "content_fixture.h"
#include "tide/stream.h"
#include "portable_torch/runtime.hpp"
#include "../../cpp/bench/streaming.h"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <iostream>
#include <limits>

namespace {
using namespace tide;
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
template<class F> void rejects(F f,const char* why) {
  bool rejected=false;try{f();}catch(const std::exception&){rejected=true;}require(rejected,why);
}
void check(at::Device device) {
  at::NoGradGuard guard;Index windows=0;
  for(int shape=0;shape<4;++shape)for(bool prefill:{false,true})for(bool device_input:{false,true}) {
    auto f=device_online::test::fixture(shape,1);ResidentLimits limits;
    limits.queue=96;limits.arrivals=128;limits.outputs=256;limits.trace=1024;limits.prefill=prefill;
    ResidentSession session(f.graph,f.model,f.initial,device,limits);Streaming oracle(f.graph,f.model,{});
    auto q=f.initial;Index start=q.cut;
    for(Index distance:{2,6,11}) {
      const auto stop=f.initial.cut+distance;std::vector<External> inputs;
      for(const auto& x:f.input)if(start<=x.time&&x.time<stop)inputs.push_back(x);
      const auto expected=oracle.run(q,inputs,stop,stop);q=expected.continuation;
      if(device_input)for(auto& x:inputs)x.value=x.value.to(device);
      const auto view=session.advance(inputs,stop,stop);
      require(view.values.device()==device&&view.coordinates.scalar_type()==at::kLong,"resident output escaped device/int64");
      require(session.cut()==stop,"resident cut not advanced");
      tide_bench::compare(session.result(),expected,true,at::kFloat);
      auto poison=session.snapshot();for(auto& [_,s]:poison.states)s.value.fill_(std::numeric_limits<float>::quiet_NaN());
      start=stop;++windows;
    }
    auto bad=std::vector<External>{{0,0,0,start,at::ones({3},at::kFloat)}};
    rejects([&]{session.advance(bad,start+1,start+1);},"invalid ledger accepted");
    require(session.cut()==start,"invalid input changed cut");
    auto reference=oracle.run(q,{},start+3,start+3);session.advance({},start+3,start+3);
    tide_bench::compare(session.result(),reference,true,at::kFloat);++windows;
    f.model.nodes[0].bias.add_(1);
    rejects([&]{session.advance({},start+4,start+4);},"stale parameter snapshot accepted");
    session.snapshot(); // May recover state before rebuilding with changed weights.
    session.close();session.close();
    rejects([&]{session.advance({},start+4,start+4);},"closed session ran");
    rejects([&]{session.snapshot();},"closed session exported state");
  }
  auto f=device_online::test::fixture(0,0);
  {at::AutoGradMode enabled(true);rejects([&]{ResidentSession s(f.graph,f.model,f.initial,device);},"implicit autograd accepted");}
  ResidentLimits small;small.stages=1;
  ResidentSession failed(f.graph,f.model,f.initial,device,small);
  rejects([&]{failed.advance(f.input,11,11);},"device stage refusal lost");
  rejects([&]{failed.snapshot();},"partial state presented as complete");failed.close();
  std::cout<<"public-resident: passed windows="<<windows<<" input_devices=cpu,npu lifecycle=true failures=true\n";
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);
    if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("resident check requires explicit NPU FP32");
    const auto device=portable_torch::resolve_device(args);
    at::set_num_threads(1);at::set_num_interop_threads(1);check(device);runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
