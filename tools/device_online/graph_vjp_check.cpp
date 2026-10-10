#include "device_backend.h"
#include "graph_vjp_fixture.h"
#include "precision_graph_fixture.h"
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
void check(at::Device device,int shape,int variant,bool prefill,int mode,at::ScalarType payload,Index feature_width=3,std::string emit="hard") {
  at::NoGradGuard guard;auto f=test::graph_vjp_fixture(shape,variant,feature_width);const auto base=f.initial.cut;
  if(emit!="hard") {
    for(auto& r:f.graph.regions)r.read_mode=shape==0?"content":shape==2?"old":"proposal";
    for(size_t n=0;n<f.graph.nodes.size();++n)if(n%2&&!f.graph.nodes[n].identity)f.graph.nodes[n].readout="norm-fp32-v1";
    f.graph.compile();f.initial.identity=f.graph.identity;
  }
  const bool half=payload==at::kHalf;const float scale=half?256.f:1.f;test::fixture_dtype(f,payload);
  ContentLimits limits;limits.prefill=prefill;limits.trace=512;limits.full_chunk_rows=3;
  limits.mode=emit;
  if(feature_width>3)limits.workspace_bytes=512*1024*1024;
  ContentFlow flow(f.graph,f.model,f.initial,device,limits);const bool warm=variant==1;
  if(warm){std::vector<External> first;for(const auto& x:f.input)if(x.time<base+2)first.push_back(x);flow.advance_device(first,base+2);}
  std::vector<External> input;for(const auto& x:f.input)if(!warm||x.time>=base+2)input.push_back(x);
  flow.advance_device(input,base+11);auto t=flow.reverse_tape();const auto width=t.full.width,nodes=int64_t(f.graph.nodes.size());
  const auto opts=t.fiber_values.options(),booleans=opts.dtype(at::kBool);
  GraphCotangents roots{at::full(t.outputs.values.sizes(),mode==4?0.f:.0625f*scale,opts),at::zeros_like(t.outputs.valid),
    at::full(t.pending.values.sizes(),.015625f*scale,opts),at::zeros_like(t.pending.valid),
    at::full({t.state.samples,nodes,width},.03125f*scale,opts),at::zeros({t.state.samples,nodes},booleans)};
  if(mode==1||mode==4||mode==5)roots.outputs_connected.copy_(t.outputs.valid);
  if(mode==3||mode==5)roots.pending_connected.copy_(t.pending.valid);
  if(mode==2||mode==5)roots.final_connected.fill_(true);
  const auto poison=std::numeric_limits<float>::quiet_NaN();
  roots.outputs.masked_fill_(roots.outputs_connected.logical_not().unsqueeze(1),poison);
  roots.pending.masked_fill_(roots.pending_connected.logical_not().unsqueeze(1),poison);
  roots.final.masked_fill_(roots.final_connected.logical_not().unsqueeze(2),poison);
  auto error=at::zeros({1},opts.dtype(at::kInt));DeviceProgram p(device);p.limit_workspace(64*1024*1024);
  auto out=append_graph_vjp(p,t,roots,error,prefill?5:1,128*1024*1024);p.finish();
  portable_torch::synchronize(device);p.run();require(!error.cpu().item<int>(),"valid graph backward refused");
  require(out.reverse_stages.cpu().item<Index>()==out.links.stages.cpu().item<Index>(),"graph reverse stage progression incomplete");
  for(auto dtype:{at::kFloat,at::kDouble}) {
    Options options;options.mode=emit;auto expected=test::graph_reference_precision(f,mode,warm,dtype,half,options);
    if(dtype==at::kFloat)tide_bench::compare(flow.result(),expected.result,true,payload,std::nullopt,half?2e-2:1e-5,half?2e-3:1e-6);
    test::compare_graph_vjp_precision(t,out,expected,half);
  }
  // Reusing the same tape and program must reset rather than accumulate a
  // previous gradient. This is replay, not a retained second-order graph.
  portable_torch::synchronize(device);p.run();require(!error.cpu().item<int>(),"graph reverse replay refused");
  Options options;options.mode=emit;
  test::compare_graph_vjp_precision(t,out,test::graph_reference_precision(f,mode,warm,at::kFloat,half,options),half);
}
void boundaries(at::Device device,at::ScalarType payload) {
  at::NoGradGuard guard;auto f=test::graph_vjp_fixture(0,0);test::fixture_dtype(f,payload);ContentFlow flow(f.graph,f.model,f.initial,device);
  flow.advance_device({},0);auto t=flow.reverse_tape();auto opts=t.fiber_values.options();
  GraphCotangents roots{at::zeros(t.outputs.values.sizes(),opts),at::zeros_like(t.outputs.valid),at::zeros(t.pending.values.sizes(),opts),at::zeros_like(t.pending.valid),
    at::ones({2,4,3},opts),at::zeros({2,4},opts.dtype(at::kBool))};
  roots.final_connected[0][0].fill_(true);roots.final[0][0].zero_(); // connected zero, other rows are None
  auto error=at::zeros({1},opts.dtype(at::kInt));DeviceProgram p(device);
  auto out=append_graph_vjp(p,t,roots,error,3,128*1024*1024);p.finish();portable_torch::synchronize(device);p.run();
  require(!error.cpu().item<int>()&&out.reverse_stages.cpu().item<Index>()==0,"empty graph backward did work");
  require(at::equal(out.initial_connected.cpu(),roots.final_connected.cpu())&&!out.initial.cpu().any().item<bool>(),"empty graph changed None/zero roots");
  require(!out.scale_connected.cpu().any().item<bool>()&&!out.full_connected.cpu().any().item<bool>()
    &&!out.decay_connected.cpu().any().item<bool>()&&!out.retention_connected.cpu().any().item<bool>(),"empty graph fabricated parameter gradients");
  bool refused=false;try{DeviceProgram small(device);append_graph_vjp(small,t,roots,error,3,1);}catch(const std::invalid_argument&){refused=true;}
  require(refused,"graph gradient budget refusal missing");
  refused=false;auto wrong=roots;wrong.outputs=roots.outputs.to(at::kHalf);
  try{DeviceProgram typed(device);append_graph_vjp(typed,t,wrong,error,3,128*1024*1024);}catch(const std::invalid_argument&){refused=true;}
  require(refused,"graph backward silently changed cotangent dtype");
  flow.advance_device(f.input,11);t=flow.reverse_tape();auto malformed=t;malformed.state.metadata=t.state.metadata.clone();
  malformed.state.metadata[0][12].fill_(1);DeviceProgram bad(device);auto rejected=append_graph_vjp(bad,malformed,roots,error,3,128*1024*1024);bad.finish();
  portable_torch::synchronize(device);bad.run();require(error.cpu().item<int>()==2,"malformed graph tape accepted");
  require(!rejected.initial_connected.cpu().any().item<bool>()&&!rejected.message_connected.cpu().any().item<bool>()
    &&!rejected.initial.cpu().any().item<bool>()&&!rejected.scales.cpu().any().item<bool>(),"failed graph preflight exposed gradients");
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||(args.dtype!=at::kFloat&&args.dtype!=at::kHalf))throw std::invalid_argument("graph VJP gate requires explicit NPU FP32/FP16");
    args.allow_npu_float16=true;
    const auto device=portable_torch::resolve_device(args);if(device.type()!=tide::device_online::resident_device_type)throw std::invalid_argument("graph VJP gate requires NPU");
    at::set_num_threads(1);at::set_num_interop_threads(1);int cases=0;
    for(int shape=0;shape<4;++shape)for(int variant=0;variant<2;++variant)for(bool prefill:{false,true})for(int mode=0;mode<6;++mode) {
      try{check(device,shape,variant,prefill,mode,args.dtype);++cases;}
      catch(...){std::cerr<<"graph VJP shape="<<shape<<" variant="<<variant<<" prefill="<<prefill<<" roots="<<mode<<'\n';throw;}
    }
    for(Index width:{1,257}){try{check(device,0,0,true,5,args.dtype,width);++cases;}
      catch(...){std::cerr<<"graph VJP feature width="<<width<<'\n';throw;}}
    for(const std::string emit:{"hst","softp"})for(int shape:{0,2,3})for(bool prefill:{false,true})for(int mode:{4,5}) {
      try{check(device,shape,1,prefill,mode,args.dtype,3,emit);++cases;}
      catch(...){std::cerr<<"graph control shape="<<shape<<" prefill="<<prefill<<" roots="<<mode<<" mode="<<emit<<'\n';throw;}
    }
    boundaries(device,args.dtype);
    std::cout<<"device-graph-vjp: passed windows="<<cases<<" dtype="<<args.dtype<<" CPU_references=FP32_FP64 replay=true scope=single_window_HARD_HST_SOFTP_sum_broadcast_identity_EMA_Add_tanh_not_optimizer\n";
    runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
