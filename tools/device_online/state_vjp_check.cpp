#include "state_vjp.h"
#include "content_fixture.h"
#include "portable_torch/runtime.hpp"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <torch/csrc/autograd/autograd.h>
#include <iostream>
#include <limits>

namespace {
using namespace tide;
using namespace tide::device_online;
void require(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
void same(const Tensor& value,const Tensor& connected,const Tensor& expected,const char* message) {
  if(connected.item<bool>()!=expected.defined()) {
    std::cerr<<message<<" connection actual="<<connected.item<bool>()<<" expected="<<expected.defined()<<'\n';
    throw std::runtime_error(message);
  }
  if(expected.defined()&&!at::allclose(value,expected.to(at::kFloat),1e-5,1e-6)) {
    std::cerr<<message<<" max_abs_error="<<(value-expected.to(at::kFloat)).abs().max().item<double>()
             <<" actual="<<value<<" expected="<<expected<<'\n';
    throw std::runtime_error(message);
  }
  else require(at::equal(value,at::zeros_like(value)),"absent VJP must not evaluate poisoned cotangents");
}
// Independent ordinary CPU autograd constructs a state chain. This tests a
// local VJP, not a CPU routing prepass supplied to a graph candidate.
void synthetic(at::Device device,Index width,int mode,at::ScalarType dtype) {
  at::AutoGradMode grad(true);
  constexpr Index samples=2,nodes=4,count=32,capacity=37;
  const Index epoch=(Index(1)<<55)+12;
  auto opts=at::TensorOptions().dtype(dtype);
  auto config=at::zeros({nodes,3},at::kLong),meta=at::zeros({capacity,13},at::kLong);
  auto values=at::full({capacity,5*width+2},std::numeric_limits<float>::quiet_NaN(),at::kFloat);
  auto cot=at::full({capacity,5,width},std::numeric_limits<float>::quiet_NaN(),at::kFloat);
  auto connected=at::zeros({capacity,5},at::kBool),fc=at::zeros({samples,nodes},at::kBool);
  auto final=at::full({samples,nodes,width},std::numeric_limits<float>::quiet_NaN(),at::kFloat);
  std::vector<Tensor> weights,initial,state,contents,losses;
  std::vector<Index> times(samples*nodes,epoch-3),observations(samples*nodes,epoch);
  for(Index n=0;n<nodes;++n) {
    config[n][0].fill_(n%2);config[n][1].fill_(n/2);config[n][2].fill_(n%3==0);
    weights.push_back((at::arange(width,opts)*.01-.2).set_requires_grad(true));
  }
  for(Index key=0;key<samples*nodes;++key) {
    auto v=(at::arange(width,opts).remainder(5)*.02-.03*key).set_requires_grad(true);
    initial.push_back(v);state.push_back(v);
  }
  Index row=0;
  for(Index step=0;step<4;++step)for(Index n=0;n<nodes;++n)for(Index b=0;b<samples;++b,++row) {
    const auto key=b*nodes+n,time=epoch+step*3;const bool active=(b+n+step)%3!=0;
    const bool adopt=config[n][2].item<bool>()||active,clear=n/2&&active;
    auto h=(at::arange(width,opts).remainder(7)*.005+.01*(row+1)).set_requires_grad(true);contents.push_back(h);
    const auto old=state[key],proposal=n%2?weights[n].sigmoid()*old+h:old;
    const auto comparison=adopt?proposal:old,next=clear?comparison*0.0:comparison;
    const Index pt=n%2?time:times[key],pc=observations[key]+(n%2),nt=adopt?pt:times[key],nc=adopt?pc:observations[key];
    meta[row].copy_(at::tensor({b,n,time,Index(active),times[key],observations[key],pt,pc,nt,nc,nt,nc,step},at::kLong));
    std::vector<Tensor> fields{h,old,proposal,comparison,next};
    for(Index field=0;field<5;++field) {
      values[row].narrow(0,field*width,width).copy_(fields[field].detach().to(at::kFloat));
      const bool on=mode==0?false:mode==1?field==4:mode==2?(row+field)%3==0:field==2;
      if(on){const double scale=mode==3?0.:((row+field)%5-2)*.03125;
        connected[row][field].fill_(true);cot[row][field].fill_(scale);losses.push_back(fields[field].sum()*scale);}
    }
    state[key]=next;times[key]=nt;observations[key]=nc;
  }
  for(Index key=0;key<samples*nodes;++key)if(mode==1||mode==2&&key%2) {
    fc.view({-1})[key].fill_(true);final.view({samples*nodes,width})[key].fill_(.125);
    losses.push_back(state[key].sum()*.125);
  }
  std::vector<Tensor> leaves=contents;leaves.insert(leaves.end(),initial.begin(),initial.end());leaves.insert(leaves.end(),weights.begin(),weights.end());
  std::vector<Tensor> expected(leaves.size());
  if(!losses.empty())expected=torch::autograd::grad({at::stack(losses).sum()},leaves,{},false,false,true);
  at::NoGradGuard guard;
  StateTape tape{meta.to(device),values.to(device),at::full({1},count,at::kLong).to(device),config.to(device),
    at::stack(weights).detach().to(device,at::kFloat),samples};
  StateCotangents seed{cot.to(device),connected.to(device),final.to(device),fc.to(device)};
  auto error=at::zeros({1},at::TensorOptions().device(device).dtype(at::kInt));
  CannProgram program(device);auto out=append_state_vjp(program,tape,seed,error,32*1024*1024);program.finish();
  portable_torch::synchronize(device);program.run();require(error.cpu().item<int>()==0,"valid state VJP refused");
  auto h=out.content.cpu(),hc=out.content_connected.cpu(),init=out.initial.cpu().reshape({samples*nodes,width});
  auto ic=out.initial_connected.cpu().reshape({-1}),dec=out.decay.cpu().sum(0),dc=out.decay_connected.cpu().any(0);
  for(Index i=0;i<count;++i)same(h[i],hc[i],expected[i],"content VJP/connection mismatch");
  for(Index i=count;i<capacity;++i)require(!hc[i].item<bool>()&&at::equal(h[i],at::zeros_like(h[i])),"padding entered VJP");
  for(Index i=0;i<samples*nodes;++i)same(init[i],ic[i],expected[count+i],"initial-state VJP/connection mismatch");
  for(Index i=0;i<nodes;++i)same(dec[i],dc[i],expected[count+samples*nodes+i],"EMA decay VJP/connection mismatch");
  // Capacity bounds simultaneous tape rows, not the lifetime event total.
  // The same captured reverse program also handles a following empty window.
  tape.count.zero_();portable_torch::synchronize(device);program.run();
  require(error.cpu().item<int>()==0&&!out.content_connected.cpu().any().item<bool>(),"empty replay retained old event connections");
  require(at::equal(out.content.cpu(),at::zeros_like(h)),"empty replay retained old event values");
  require(at::equal(out.initial_connected.cpu(),fc),"empty replay changed final-state connectivity");
  require(!out.decay_connected.cpu().any().item<bool>(),"empty replay retained old parameter connections");
  tape.count.fill_(count);
  // Every seeded call resets output storage; malformed device metadata fails
  // without running a numerical prefix or touching independent forward state.
  if(width==1&&mode==2&&dtype==at::kFloat) {
    tape.metadata[1][0].fill_(samples);error.zero_();
    CannProgram bad(device);auto refused=append_state_vjp(bad,tape,seed,error,32*1024*1024);bad.finish();
    portable_torch::synchronize(device);bad.run();require(error.cpu().item<int>()==2,"malformed tape accepted");
    require(at::equal(refused.initial.cpu(),at::zeros_like(init).reshape({samples,nodes,width})),"failed VJP wrote output");
    bool bounded=false;try{CannProgram small(device);append_state_vjp(small,tape,seed,error,1);}catch(const std::invalid_argument&){bounded=true;}
    require(bounded,"VJP budget refusal missing");
  }
}
void actual_tape(at::Device device,bool prefill) {
  at::NoGradGuard guard;auto f=test::fixture(0,0);
  ContentLimits limits;limits.prefill=prefill;limits.trace=512;
  ContentFlow flow(f.graph,f.model,f.initial,device,limits);flow.advance_device(f.input,11);
  auto tape=flow.state_tape();const auto capacity=tape.values.size(0),width=tape.decay.size(1),nodes=tape.config.size(0);
  auto opts=tape.values.options();
  StateCotangents seed{at::zeros({capacity,5,width},opts),at::zeros({capacity,5},opts.dtype(at::kBool)),
    at::ones({tape.samples,nodes,width},opts),at::ones({tape.samples,nodes},opts.dtype(at::kBool))};
  auto error=at::zeros({1},opts.dtype(at::kInt));CannProgram reverse(device);
  auto out=append_state_vjp(reverse,tape,seed,error,32*1024*1024);reverse.finish();
  portable_torch::synchronize(device);reverse.run();require(error.cpu().item<int>()==0,"actual forward journal rejected");
  require(at::isfinite(out.initial.cpu()).all().item<bool>(),"actual tape produced nonfinite VJP");
  require(out.initial_connected.cpu().all().item<bool>(),"final state lost initial-state connection");
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);
    if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("state VJP gate requires explicit NPU FP32");
    const auto device=portable_torch::resolve_device(args);
    if(device.type()!=c10::DeviceType::PrivateUse1)throw std::invalid_argument("state VJP gate requires NPU");
    at::set_num_threads(1);at::set_num_interop_threads(1);
    int cases=0;
    for(auto dtype:{at::kFloat,at::kDouble})for(Index width:{1,7,257})for(int mode=0;mode<4;++mode) {
      try{synthetic(device,width,mode,dtype);++cases;}
      catch(...){std::cerr<<"state VJP case width="<<width<<" mode="<<mode<<" reference="<<dtype<<'\n';throw;}
    }
    actual_tape(device,false);actual_tape(device,true);
    std::cout<<"device-state-vjp: passed isolated="<<cases<<" actual_device_tapes=2 scope=identity_EMA_HARD_first_order_not_graph_training\n";
    runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
