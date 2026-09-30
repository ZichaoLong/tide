#include "content_fixture.h"
#include "portable_torch/runtime.hpp"
#include "tide/stream.h"
#include "../../cpp/bench/streaming.h"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <algorithm>
#include <iostream>
#include <limits>

namespace {
using namespace tide;
using namespace tide::device_online;
void require(bool yes,const char* why){if(!yes)throw std::runtime_error(why);}
template<class F> void rejects(F f,const char* why) {
  bool no=false;try{f();}catch(const std::exception&){no=true;}require(no,why);
}
std::vector<Output> outputs(const ContentWindow& view,const Graph& graph,at::Device device) {
  for(const auto& tensor:{view.outputs.coordinates,view.outputs.values,view.outputs.valid,
      view.output_stats,view.pending_stats,view.stages,view.events,view.full_chunks})
    require(tensor.device()==device,"window buffer escaped the NPU");
  auto coords=view.outputs.coordinates.cpu(),values=view.outputs.values.cpu(),live=view.outputs.valid.cpu();
  std::vector<Output> out;
  auto c=coords.accessor<Index,2>();
  for(Index i=0;i<live.numel();++i)if(live[i].item<bool>())out.push_back({c[i][0],c[i][2],c[i][4],values[i].clone()});
  std::sort(out.begin(),out.end(),[&](const Output& a,const Output& b){return
    std::tie(a.time,a.batch,graph.outputs[a.port],a.port)<std::tie(b.time,b.batch,graph.outputs[b.port],b.port);});
  require(Index(out.size())==view.output_stats.cpu()[0].item<Index>(),"device output length differs");
  return out;
}
void check(at::Device device,bool without_diagnostics) {
  at::NoGradGuard guard;Index cases=0;
  const auto diagnostics=without_diagnostics?std::vector<bool>{false}:std::vector<bool>{false,true};
  for(int shape=0;shape<4;++shape)for(int variant=0;variant<2;++variant)
  for(bool prefill:{false,true})for(const std::string mode:{"content","old","proposal"})for(bool trace:diagnostics) {
    auto f=test::fixture(shape,variant);
    for(auto& r:f.graph.regions)r.read_mode=mode;
    for(auto& n:f.graph.nodes)if(!n.identity)n.full="tanh";
    f.graph.compile();f.initial.identity=f.graph.identity;
    ContentLimits limits;limits.queue=96;limits.arrivals=128;limits.outputs=256;limits.trace=trace?1024:0;
    limits.prefill=prefill;limits.diagnostics=trace;limits.full_chunk_rows=3;
    ContentFlow flow(f.graph,f.model,f.initial,device,limits);Streaming oracle(f.graph,f.model,{});
    Result initial,expected_initial;initial.continuation=flow.snapshot();expected_initial.continuation=f.initial;
    tide_bench::compare(initial,expected_initial,false,at::kFloat);
    auto q=f.initial;Index previous=q.cut;Result reference;
    for(Index step:{2,6,11}) {
      const Index until=f.initial.cut+step;std::vector<External> input;
      for(const auto& x:f.input)if(x.time>=previous&&x.time<until)input.push_back(x);
      reference=oracle.run(q,input,until,until);q=reference.continuation;
      auto view=flow.advance_device(input,until);auto actual_outputs=outputs(view,f.graph,device);
      // No state/history/pending snapshot is requested between these windows.
      Result a,b;a.outputs=actual_outputs;b.outputs=reference.outputs;
      a.continuation=b.continuation=f.initial;
      tide_bench::compare(a,b,false,at::kFloat);
      require(view.events.cpu().item<Index>()==Index(reference.trace.size()),"device event count differs");
      require(view.pending_stats.cpu()[0].item<Index>()==Index(q.pending.size()),"device pending length differs");
      previous=until;++cases;
    }
    auto actual=flow.result();tide_bench::compare(actual,reference,trace,at::kFloat);
    require(actual.stats.at("diagnostics")==trace,"diagnostic mode changed");
    if(!trace)require(actual.trace.empty()&&actual.messages.empty(),"disabled journals were materialized");
    // Exported CPU data must never become the continuation used by the device.
    auto snapshot=flow.snapshot();
    for(auto& [_,state]:snapshot.states)state.value.fill_(std::numeric_limits<float>::quiet_NaN());
    for(auto& atom:snapshot.pending)atom.value.fill_(std::numeric_limits<float>::quiet_NaN());
    snapshot.history.clear();
    reference=oracle.run(q,{},previous+3,previous+3);
    flow.advance_device({},previous+3);tide_bench::compare(flow.result(),reference,trace,at::kFloat);++cases;
  }
  auto f=test::fixture(0,0);ContentLimits l;l.diagnostics=false;l.trace=0;l.stages=1;
  ContentFlow failed(f.graph,f.model,f.initial,device,l);
  rejects([&]{failed.advance_device(f.input,11);},"window execution failure silently succeeded");
  rejects([&]{failed.snapshot();},"failed window exported a complete cut");
  rejects([&]{failed.result();},"failed window exported a result");
  rejects([&]{failed.advance_device({},11);},"failed device window resumed");
  std::cout<<"device-window: passed cases="<<cases<<" diagnostics="<<(without_diagnostics?"off":"both")
    <<" delayed_export=true snapshot_isolated=true failed_snapshot_refused=true\n";
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    bool lean=false;std::vector<char*> cli{argv[0]};
    for(int i=1;i<argc;++i)if(std::string(argv[i])=="--without-diagnostics")lean=true;else cli.push_back(argv[i]);
    auto args=portable_torch::parse_cli(cli.size(),cli.data(),true);
    if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("window check requires explicit NPU FP32");
    auto device=portable_torch::resolve_device(args);
    if(device.type()!=c10::DeviceType::PrivateUse1)throw std::invalid_argument("window check requires NPU");
    at::set_num_threads(1);at::set_num_interop_threads(1);check(device,lean);runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
