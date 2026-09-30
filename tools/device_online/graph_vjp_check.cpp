#include "graph_vjp_fixture.h"
#include "portable_torch/runtime.hpp"
#include "../../cpp/bench/streaming.h"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <iostream>
#include <limits>

namespace {
using namespace tide;
using namespace tide::device_online;
void require(bool on,const char* message){if(!on)throw std::runtime_error(message);}
void check(at::Device device,int shape,int variant,bool prefill,int mode,Index feature_width=3) {
  at::NoGradGuard guard;auto f=test::graph_vjp_fixture(shape,variant,feature_width);const auto base=f.initial.cut;
  ContentLimits limits;limits.prefill=prefill;limits.trace=512;limits.full_chunk_rows=3;
  if(feature_width>3)limits.workspace_bytes=512*1024*1024;
  ContentFlow flow(f.graph,f.model,f.initial,device,limits);const bool warm=variant==1;
  if(warm){std::vector<External> first;for(const auto& x:f.input)if(x.time<base+2)first.push_back(x);flow.advance_device(first,base+2);}
  std::vector<External> input;for(const auto& x:f.input)if(!warm||x.time>=base+2)input.push_back(x);
  flow.advance_device(input,base+11);auto t=flow.reverse_tape();const auto width=t.full.width,nodes=int64_t(f.graph.nodes.size());
  const auto opts=t.fiber_values.options(),booleans=opts.dtype(at::kBool);
  GraphCotangents roots{at::full_like(t.outputs.values,mode==4?0.f:.0625f),at::zeros_like(t.outputs.valid),
    at::full_like(t.pending.values,.015625f),at::zeros_like(t.pending.valid),
    at::full({t.state.samples,nodes,width},.03125f,opts),at::zeros({t.state.samples,nodes},booleans)};
  if(mode==1||mode==4||mode==5)roots.outputs_connected.copy_(t.outputs.valid);
  if(mode==3||mode==5)roots.pending_connected.copy_(t.pending.valid);
  if(mode==2||mode==5)roots.final_connected.fill_(true);
  const auto poison=std::numeric_limits<float>::quiet_NaN();
  roots.outputs.masked_fill_(roots.outputs_connected.logical_not().unsqueeze(1),poison);
  roots.pending.masked_fill_(roots.pending_connected.logical_not().unsqueeze(1),poison);
  roots.final.masked_fill_(roots.final_connected.logical_not().unsqueeze(2),poison);
  auto error=at::zeros({1},opts.dtype(at::kInt));CannProgram p(device);p.limit_workspace(64*1024*1024);
  auto out=append_graph_vjp(p,t,roots,error,prefill?5:1,128*1024*1024);p.finish();
  portable_torch::synchronize(device);p.run();require(!error.cpu().item<int>(),"valid graph backward refused");
  require(out.reverse_stages.cpu().item<Index>()==out.links.stages.cpu().item<Index>(),"graph reverse stage progression incomplete");
  for(auto dtype:{at::kFloat,at::kDouble}) {
    auto expected=test::graph_reference(f,mode,warm,dtype);
    if(dtype==at::kFloat)tide_bench::compare(flow.result(),expected.result,true,at::kFloat);
    test::compare_graph_vjp(t,out,expected);
  }
  // Reusing the same tape and program must reset rather than accumulate a
  // previous gradient. This is replay, not a retained second-order graph.
  portable_torch::synchronize(device);p.run();require(!error.cpu().item<int>(),"graph reverse replay refused");
  test::compare_graph_vjp(t,out,test::graph_reference(f,mode,warm,at::kFloat));
}
void boundaries(at::Device device) {
  at::NoGradGuard guard;auto f=test::graph_vjp_fixture(0,0);ContentFlow flow(f.graph,f.model,f.initial,device);
  flow.advance_device({},0);auto t=flow.reverse_tape();auto opts=t.fiber_values.options();
  GraphCotangents roots{at::zeros_like(t.outputs.values),at::zeros_like(t.outputs.valid),at::zeros_like(t.pending.values),at::zeros_like(t.pending.valid),
    at::ones({2,4,3},opts),at::zeros({2,4},opts.dtype(at::kBool))};
  roots.final_connected[0][0].fill_(true);roots.final[0][0].zero_(); // connected zero, other rows are None
  auto error=at::zeros({1},opts.dtype(at::kInt));CannProgram p(device);
  auto out=append_graph_vjp(p,t,roots,error,3,128*1024*1024);p.finish();portable_torch::synchronize(device);p.run();
  require(!error.cpu().item<int>()&&out.reverse_stages.cpu().item<Index>()==0,"empty graph backward did work");
  require(at::equal(out.initial_connected.cpu(),roots.final_connected.cpu())&&!out.initial.cpu().any().item<bool>(),"empty graph changed None/zero roots");
  require(!out.scale_connected.cpu().any().item<bool>()&&!out.full_connected.cpu().any().item<bool>()
    &&!out.decay_connected.cpu().any().item<bool>()&&!out.retention_connected.cpu().any().item<bool>(),"empty graph fabricated parameter gradients");
  bool refused=false;try{CannProgram small(device);append_graph_vjp(small,t,roots,error,3,1);}catch(const std::invalid_argument&){refused=true;}
  require(refused,"graph gradient budget refusal missing");
  refused=false;auto wrong=roots;wrong.outputs=roots.outputs.to(at::kHalf);
  try{CannProgram typed(device);append_graph_vjp(typed,t,wrong,error,3,128*1024*1024);}catch(const std::invalid_argument&){refused=true;}
  require(refused,"graph backward silently changed cotangent dtype");
  flow.advance_device(f.input,11);t=flow.reverse_tape();auto malformed=t;malformed.state.metadata=t.state.metadata.clone();
  malformed.state.metadata[0][12].fill_(1);CannProgram bad(device);auto rejected=append_graph_vjp(bad,malformed,roots,error,3,128*1024*1024);bad.finish();
  portable_torch::synchronize(device);bad.run();require(error.cpu().item<int>()==2,"malformed graph tape accepted");
  require(!rejected.initial_connected.cpu().any().item<bool>()&&!rejected.message_connected.cpu().any().item<bool>()
    &&!rejected.initial.cpu().any().item<bool>()&&!rejected.scales.cpu().any().item<bool>(),"failed graph preflight exposed gradients");
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("graph VJP gate requires explicit NPU FP32");
    const auto device=portable_torch::resolve_device(args);if(device.type()!=c10::DeviceType::PrivateUse1)throw std::invalid_argument("graph VJP gate requires NPU");
    at::set_num_threads(1);at::set_num_interop_threads(1);int cases=0;
    for(int shape=0;shape<4;++shape)for(int variant=0;variant<2;++variant)for(bool prefill:{false,true})for(int mode=0;mode<6;++mode) {
      try{check(device,shape,variant,prefill,mode);++cases;}
      catch(...){std::cerr<<"graph VJP shape="<<shape<<" variant="<<variant<<" prefill="<<prefill<<" roots="<<mode<<'\n';throw;}
    }
    for(Index width:{1,257}){try{check(device,0,0,true,5,width);++cases;}
      catch(...){std::cerr<<"graph VJP feature width="<<width<<'\n';throw;}}
    boundaries(device);
    std::cout<<"device-graph-vjp: passed windows="<<cases<<" CPU_references=FP32_FP64 replay=true scope=single_window_HARD_sum_broadcast_identity_EMA_Add_tanh_not_optimizer\n";
    runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
