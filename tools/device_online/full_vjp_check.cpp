#include "full_vjp_fixture.h"
#include "content_fixture.h"
#include "portable_torch/runtime.hpp"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <iostream>

namespace {
using namespace tide;
using namespace tide::device_online;
void require(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
FullTape upload(const FullTape& t,at::Device device) {
  return {t.metadata.to(device),t.values.to(device),t.count.to(device),t.kinds.to(device),
    t.weights.defined()?t.weights.to(device):Tensor(),t.biases.defined()?t.biases.to(device):Tensor(),t.samples,t.width,t.has_tanh};
}
void synthetic(at::Device device,Index width,int mode,bool tanh,Index chunk,at::ScalarType dtype) {
  at::NoGradGuard guard;auto f=test::full_fixture(width,mode,tanh);auto tape=upload(f.tape,device);
  auto gradient=f.gradient.to(device),connected=f.connected.to(device),error=at::zeros({1},gradient.options().dtype(at::kInt));
  CannProgram program(device);program.limit_workspace(32*1024*1024);
  auto out=append_full_vjp(program,tape,gradient,connected,error,chunk,32*1024*1024);program.finish();
  for(Index count:{19,7,0}) {
    f.tape.count.fill_(count);tape.count.copy_(f.tape.count);
    portable_torch::synchronize(device);program.run();require(!error.cpu().item<int>(),"valid Full VJP refused");
    test::full_compare(f,out,dtype);
    Index selected=0;for(Index i=0;i<count;++i)selected+=f.connected[i].item<bool>()&&f.tape.kinds[f.tape.metadata[i][1].item<Index>()].item<Index>();
    require(out.chunks.cpu().item<Index>()==(selected+out.chunk_rows-1)/out.chunk_rows,"Full VJP chunk progression mismatch");
  }
}
void refusals(at::Device device) {
  at::NoGradGuard guard;auto f=test::full_fixture(7,1,true);auto tape=upload(f.tape,device);
  auto gradient=f.gradient.to(device),connected=f.connected.to(device),error=at::zeros({1},gradient.options().dtype(at::kInt));
  auto bad=[&](int code){CannProgram p(device);auto out=append_full_vjp(p,tape,gradient,connected,error,3,1024*1024);p.finish();
    portable_torch::synchronize(device);p.run();require(error.cpu().item<int>()==code,"Full VJP missing device refusal");
    require(!out.content.cpu().any().item<bool>()&&!out.content_connected.cpu().any().item<bool>()
      &&!out.comparison.cpu().any().item<bool>()&&!out.parameter_connected.cpu().any().item<bool>(),"invalid Full tape exposed partial gradient");
    error.zero_();tape=upload(f.tape,device);connected.copy_(f.connected);};
  tape.count.fill_(24);bad(2);tape.metadata[1][0].fill_(2);bad(2);tape.metadata[1][1].fill_(-1);bad(2);
  tape.metadata[1][2].fill_(-1);bad(2);tape.metadata[1][3].fill_(2);bad(2);connected[0].fill_(true);bad(2);
  tape.kinds[1].fill_(2);bad(12);
  bool refused=false;try{CannProgram p(device);append_full_vjp(p,tape,gradient,connected,error,3,1);}catch(const std::invalid_argument&){refused=true;}
  require(refused,"Full VJP budget refusal missing");
  refused=false;try{CannProgram p(device);append_full_vjp(p,tape,gradient.to(at::kHalf),connected,error,3,1024*1024);}catch(const std::invalid_argument&){refused=true;}
  require(refused,"Full VJP silently changed requested dtype");
}
void actual_tape(at::Device device,bool prefill) {
  at::NoGradGuard guard;auto f=test::fixture(0,0);
  for(auto& n:f.graph.nodes)if(!n.identity)n.full="tanh";
  for(auto& w:f.model.nodes)w.weight.mul_(.125);
  f.graph.compile();f.initial.identity=f.graph.identity;
  ContentLimits limits;limits.prefill=prefill;limits.trace=256;limits.full_chunk_rows=3;
  ContentFlow flow(f.graph,f.model,f.initial,device,limits);flow.advance_device(f.input,11);auto tape=flow.full_tape();
  test::FullFixture local{{tape.metadata.cpu(),tape.values.cpu(),tape.count.cpu(),tape.kinds.cpu(),
    tape.weights.cpu(),tape.biases.cpu(),tape.samples,tape.width,tape.has_tanh},
    at::ones({tape.values.size(0),tape.width},at::kFloat),tape.metadata.select(1,3).cpu().eq(1)};
  auto gradient=local.gradient.to(device),connected=local.connected.to(device),error=at::zeros({1},gradient.options().dtype(at::kInt));
  CannProgram p(device);auto out=append_full_vjp(p,tape,gradient,connected,error,3,32*1024*1024);p.finish();
  portable_torch::synchronize(device);p.run();require(!error.cpu().item<int>(),"actual Full forward tape rejected");
  test::full_compare(local,out,at::kFloat);test::full_compare(local,out,at::kDouble);
  flow.close();bool refused=false;try{flow.full_tape();}catch(const std::logic_error&){refused=true;}
  require(refused,"closed owner exposed Full tape");
  limits.diagnostics=false;ContentFlow inference(f.graph,f.model,f.initial,device,limits);
  refused=false;try{inference.full_tape();}catch(const std::logic_error&){refused=true;}
  require(refused,"unrecorded forward exposed Full tape");
  f.graph.nodes[0].full="lh-relu-identity-v1";f.graph.compile();f.initial.identity=f.graph.identity;limits.diagnostics=true;
  ContentFlow unsupported(f.graph,f.model,f.initial,device,limits);
  require(unsupported.full_tape().extra.lh_kinds.defined(),"LH Full tape lost its explicit adjoint contract");
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);
    if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("Full VJP gate requires explicit NPU FP32");
    const auto device=portable_torch::resolve_device(args);
    if(device.type()!=c10::DeviceType::PrivateUse1)throw std::invalid_argument("Full VJP gate requires NPU");
    at::set_num_threads(1);at::set_num_interop_threads(1);int cases=0;
    for(auto dtype:{at::kFloat,at::kDouble})for(Index width:{1,7,257})for(int mode=0;mode<4;++mode)
    for(bool tanh:{false,true})for(Index chunk:{1,5}) {
      try{synthetic(device,width,mode,tanh,chunk,dtype);++cases;}
      catch(...){std::cerr<<"Full VJP width="<<width<<" mode="<<mode<<" tanh="<<tanh<<" chunk="<<chunk<<" reference="<<dtype<<'\n';throw;}
    }
    refusals(device);for(bool prefill:{false,true})actual_tape(device,prefill);
    std::cout<<"device-full-vjp: passed isolated="<<cases<<" actual_device_tapes=2 scope=identity_tanh_first_order_not_graph_training\n";
    runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
