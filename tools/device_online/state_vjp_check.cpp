#include "device_backend.h"
#include "state_vjp.h"
#include "content_fixture.h"
#include "portable_torch/runtime.hpp"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/custom_function.h>
#include <iostream>
#include <limits>

namespace {
using namespace tide;
using namespace tide::device_online;
void require(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
void same(const Tensor& value,const Tensor& connected,const Tensor& expected,const char* message,bool half=false) {
  if(connected.item<bool>()!=expected.defined()) {
    std::cerr<<message<<" connection actual="<<connected.item<bool>()<<" expected="<<expected.defined()<<'\n';
    throw std::runtime_error(message);
  }
  if(expected.defined()) {
    if(!at::allclose(value,expected.to(at::kFloat),half?2e-3:1e-5,half?2e-5:1e-6)) {
      std::cerr<<message<<" max_abs_error="<<(value-expected.to(at::kFloat)).abs().max().item<double>()
               <<" actual="<<value<<" expected="<<expected<<'\n';
      throw std::runtime_error(message);
    }
  } else require(at::equal(value,at::zeros_like(value)),"absent VJP must not evaluate poisoned cotangents");
}
class HalfSigmoid : public torch::autograd::Function<HalfSigmoid> {
 public:
  static Tensor forward(torch::autograd::AutogradContext* ctx,const Tensor& x) {
    auto y=x.to(at::kHalf).sigmoid().to(at::kFloat);ctx->save_for_backward({y});return y;
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list g) {
    const auto y=ctx->get_saved_variables()[0];return {g[0]*(1-y)*y};
  }
};
// Independent ordinary CPU autograd constructs a state chain. This tests a
// local VJP, not a CPU routing prepass supplied to a graph candidate.
void synthetic(at::Device device,Index width,int mode,at::ScalarType dtype,bool repeat=false,
               double rho=.5,Index chunk=256,Index period=1) {
  at::AutoGradMode grad(true);
  constexpr Index samples=2,nodes=4,count=32,capacity=37;
  const Index epoch=(Index(1)<<55)+12;
  auto opts=at::TensorOptions().dtype(dtype);
  auto leaf=[&](Tensor value){return (dtype==at::kHalf?value.to(at::kHalf).to(at::kFloat):value).set_requires_grad(true);};
  auto round=[&](const Tensor& x){return dtype==at::kHalf?x+(x.detach().to(at::kHalf).to(at::kFloat)-x.detach()):x;};
  auto config=at::zeros({nodes,3},at::kLong),meta=at::zeros({capacity,13},at::kLong);
  auto clocks=at::tensor({1,0,1},at::kLong).repeat({nodes,1});
  if(repeat&&period==3)clocks[2].copy_(at::tensor({3,2,1},at::kLong));
  auto values=at::full({capacity,5*width+2},std::numeric_limits<float>::quiet_NaN(),at::kFloat);
  auto cot=at::full({capacity,5,width},std::numeric_limits<float>::quiet_NaN(),at::kFloat);
  auto connected=at::zeros({capacity,5},at::kBool),fc=at::zeros({samples,nodes},at::kBool);
  auto final=at::full({samples,nodes,width},std::numeric_limits<float>::quiet_NaN(),at::kFloat);
  std::vector<Tensor> weights,retentions,initial,state,contents,losses;
  std::vector<Index> times(samples*nodes,epoch-3),observations(samples*nodes,epoch);
  for(Index n=0;n<nodes;++n) {
    config[n][0].fill_(repeat&&n==2?2:n%2);config[n][1].fill_(n/2);config[n][2].fill_(n%3==0);
    weights.push_back(leaf(at::arange(width,opts)*.01-.2));
    retentions.push_back(leaf(at::full({},rho,opts)));
  }
  for(Index key=0;key<samples*nodes;++key) {
    auto v=leaf(at::arange(width,opts).remainder(5)*.02-.03*key);
    initial.push_back(v);state.push_back(v);
  }
  Index row=0;
  for(Index step=0;step<4;++step)for(Index n=0;n<nodes;++n)for(Index b=0;b<samples;++b,++row) {
    const auto key=b*nodes+n,time=epoch+step*3;const bool active=(b+n+step)%3!=0;
    const bool adopt=config[n][2].item<bool>()||active,clear=n/2&&active;
    auto h=leaf(at::arange(width,opts).remainder(7)*.005+.01*(row+1));contents.push_back(h);
    const Index kind=config[n][0].item<Index>();
    const auto old=state[key];auto proposal=old;
    if(kind==1){auto coefficient=dtype==at::kHalf?HalfSigmoid::apply(weights[n]):weights[n].sigmoid();proposal=round(round(coefficient*old)+h);}
    if(kind==2){for(Index tick=0;tick<(time-times[key])/period;++tick)proposal=round(proposal*retentions[n]);proposal=round(h+proposal);}
    const auto comparison=adopt?proposal:old,next=clear?comparison*0.0:comparison;
    const Index pt=kind?time:times[key],pc=observations[key]+bool(kind),nt=adopt?pt:times[key],nc=adopt?pc:observations[key];
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
  leaves.insert(leaves.end(),retentions.begin(),retentions.end());
  std::vector<Tensor> expected(leaves.size());
  if(!losses.empty())expected=torch::autograd::grad({at::stack(losses).sum()},leaves,{},false,false,true);
  at::NoGradGuard guard;
  StateTape tape{meta.to(device),values.to(device),at::full({1},count,at::kLong).to(device),config.to(device),
    at::stack(weights).detach().to(device,dtype==at::kHalf?at::kHalf:at::kFloat),
    at::stack(retentions).detach().to(device,dtype==at::kHalf?at::kHalf:at::kFloat),
    clocks.to(device),samples,repeat,16};
  StateCotangents seed{cot.to(device),connected.to(device),final.to(device),fc.to(device)};
  auto error=at::zeros({1},at::TensorOptions().device(device).dtype(at::kInt));
  DeviceProgram program(device);auto out=append_state_vjp(program,tape,seed,error,32*1024*1024,chunk);program.finish();
  portable_torch::synchronize(device);program.run();require(error.cpu().item<int>()==0,"valid state VJP refused");
  auto h=out.content.cpu(),hc=out.content_connected.cpu(),init=out.initial.cpu().reshape({samples*nodes,width});
  auto ic=out.initial_connected.cpu().reshape({-1}),dec=out.decay.cpu().sum(0),dc=out.decay_connected.cpu().any(0);
  for(Index i=0;i<count;++i)same(h[i],hc[i],expected[i],"content VJP/connection mismatch",dtype==at::kHalf);
  for(Index i=count;i<capacity;++i)require(!hc[i].item<bool>()&&at::equal(h[i],at::zeros_like(h[i])),"padding entered VJP");
  for(Index i=0;i<samples*nodes;++i)same(init[i],ic[i],expected[count+i],"initial-state VJP/connection mismatch",dtype==at::kHalf);
  for(Index i=0;i<nodes;++i)same(dec[i],dc[i],expected[count+samples*nodes+i],"EMA decay VJP/connection mismatch",dtype==at::kHalf);
  auto ret=out.retention_components.cpu().sum(std::vector<int64_t>{0,2}),rc=out.retention_connected.cpu().any(0);
  for(Index i=0;i<nodes;++i)same(ret[i],rc[i],expected[count+samples*nodes+nodes+i],"Add retention VJP/connection mismatch",dtype==at::kHalf);
  // Capacity bounds simultaneous tape rows, not the lifetime event total.
  // The same captured reverse program also handles a following empty window.
  tape.count.zero_();portable_torch::synchronize(device);program.run();
  require(error.cpu().item<int>()==0&&!out.content_connected.cpu().any().item<bool>(),"empty replay retained old event connections");
  require(at::equal(out.content.cpu(),at::zeros_like(h)),"empty replay retained old event values");
  require(at::equal(out.initial_connected.cpu(),fc),"empty replay changed final-state connectivity");
  require(!out.decay_connected.cpu().any().item<bool>(),"empty replay retained old parameter connections");
  require(!out.retention_connected.cpu().any().item<bool>(),"empty replay retained old retention connections");
  tape.count.fill_(count);
  // Every seeded call resets output storage; malformed device metadata fails
  // without running a numerical prefix or touching independent forward state.
  if(!repeat&&width==1&&mode==2&&dtype==at::kFloat) {
    tape.metadata[1][0].fill_(samples);error.zero_();
    DeviceProgram bad(device);auto refused=append_state_vjp(bad,tape,seed,error,32*1024*1024);bad.finish();
    portable_torch::synchronize(device);bad.run();require(error.cpu().item<int>()==2,"malformed tape accepted");
    require(at::equal(refused.initial.cpu(),at::zeros_like(init).reshape({samples,nodes,width})),"failed VJP wrote output");
    bool bounded=false;try{DeviceProgram small(device);append_state_vjp(small,tape,seed,error,1);}catch(const std::invalid_argument&){bounded=true;}
    require(bounded,"VJP budget refusal missing");
  }
  if(repeat&&width==1&&mode==2&&dtype==at::kFloat&&rho==0&&chunk==2&&period==1) {
    tape.max_repeat_ticks=1;DeviceProgram bounded(device);error.zero_();
    auto refused=append_state_vjp(bounded,tape,seed,error,32*1024*1024,chunk);bounded.finish();
    portable_torch::synchronize(device);bounded.run();require(error.cpu().item<int>()==8,"repeat work bound missing");
    require(!refused.content_connected.cpu().any().item<bool>(),"bounded refusal exposed partial adjoints");
  }
}
void actual_tape(at::Device device,bool prefill,bool repeat,at::ScalarType payload) {
  at::NoGradGuard guard;auto f=test::fixture(0,0);
  if(repeat){f.graph.nodes[0].memory="lh-add-repeat-v1";f.model.nodes[0].extra["add_retention"]=at::full({},.99123f,at::kFloat);}
  f.graph.compile();f.initial.identity=f.graph.identity;
  test::model_dtype(f.model,payload);
  for(auto& [_,s]:f.initial.states)s.value=s.value.to(payload);
  for(auto& x:f.input)x.value=x.value.to(payload);
  ContentLimits limits;limits.prefill=prefill;limits.trace=512;
  ContentFlow flow(f.graph,f.model,f.initial,device,limits);flow.advance_device(f.input,11);
  auto tape=flow.state_tape();const auto capacity=tape.values.size(0),width=tape.decay.size(1),nodes=tape.config.size(0);
  auto opts=tape.values.options();
  StateCotangents seed{at::zeros({capacity,5,width},opts),at::zeros({capacity,5},opts.dtype(at::kBool)),
    at::ones({tape.samples,nodes,width},opts),at::ones({tape.samples,nodes},opts.dtype(at::kBool))};
  auto error=at::zeros({1},opts.dtype(at::kInt));DeviceProgram reverse(device);
  auto out=append_state_vjp(reverse,tape,seed,error,32*1024*1024);reverse.finish();
  portable_torch::synchronize(device);reverse.run();require(error.cpu().item<int>()==0,"actual forward journal rejected");
  require(at::isfinite(out.initial.cpu()).all().item<bool>(),"actual tape produced nonfinite VJP");
  require(out.initial_connected.cpu().all().item<bool>(),"final state lost initial-state connection");
}
// Independent quantized forward arithmetic with the usual identity cast VJP,
// lifted to FP32 so this anchor separates saved-value rounding from reverse
// accumulator precision. A whole-forward FP32 replay fails this case.
void rounding_sensitive(at::Device device,Index chunk) {
  constexpr Index width=7,ticks=1024,capacity=3,epoch=(Index(1)<<55);
  auto reference=[&](bool rounded) {
    at::AutoGradMode grad(true);
    auto old=at::full({width},.75f,at::kFloat).set_requires_grad(true);
    auto rho=at::full({},.99951171875f,at::kFloat).set_requires_grad(true);
    auto value=old;
    for(Index tick=0;tick<ticks;++tick) {
      const auto product=value*rho;
      value=rounded?product+(product.detach().to(at::kHalf).to(at::kFloat)-product.detach()):product;
    }
    return torch::autograd::grad({value.sum()},std::vector<Tensor>{old,rho});
  };
  const auto expected=reference(true),wrong=reference(false);
  require((expected[1]-wrong[1]).abs().item<float>()>1.f,"rounding anchor no longer distinguishes FP32 replay");
  at::NoGradGuard guard;
  auto meta=at::zeros({capacity,13},at::kLong);
  meta[0].copy_(at::tensor(std::vector<Index>{0,0,epoch+ticks,1,epoch,0,epoch+ticks,1,epoch+ticks,1,epoch+ticks,1,0},at::kLong));
  auto values=at::full({capacity,5*width+2},std::numeric_limits<float>::quiet_NaN(),at::kFloat);
  values[0].narrow(0,width,width).fill_(.75f);
  StateTape tape{meta.to(device),values.to(device),at::ones({1},at::kLong).to(device),
    at::tensor({2,0,1},at::kLong).reshape({1,3}).to(device),at::zeros({1,width},at::kHalf).to(device),
    at::full({1},.99951171875f,at::kHalf).to(device),at::tensor({1,0,1},at::kLong).reshape({1,3}).to(device),1,true,ticks};
  auto floats=values.options().device(device),bits=floats.dtype(at::kBool);
  StateCotangents cot{at::full({capacity,5,width},std::numeric_limits<float>::quiet_NaN(),floats),
    at::zeros({capacity,5},bits),at::ones({1,1,width},floats),at::ones({1,1},bits)};
  auto error=at::zeros({1},floats.dtype(at::kInt));DeviceProgram p(device);
  const auto out=append_state_vjp(p,tape,cot,error,1024*1024,chunk);p.finish();
  portable_torch::synchronize(device);p.run();require(!error.cpu().item<int>(),"quantized Add anchor refused");
  same(out.initial.cpu()[0][0],out.initial_connected.cpu()[0][0],expected[0],"quantized Add initial adjoint");
  same(out.retention_components.cpu().sum(),out.retention_connected.cpu().any(),expected[1],"quantized Add retention adjoint");
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);
    if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||(args.dtype!=at::kFloat&&args.dtype!=at::kHalf))throw std::invalid_argument("state VJP gate requires explicit NPU FP32/FP16");
    args.allow_npu_float16=true;
    const auto device=portable_torch::resolve_device(args);
    if(device.type()!=tide::device_online::resident_device_type)throw std::invalid_argument("state VJP gate requires NPU");
    at::set_num_threads(1);at::set_num_interop_threads(1);
    int cases=0;
    const std::vector<at::ScalarType> references=args.dtype==at::kHalf?std::vector<at::ScalarType>{at::kHalf}:
      std::vector<at::ScalarType>{at::kFloat,at::kDouble};
    for(auto dtype:references)for(Index width:{1,7,257})for(int mode=0;mode<4;++mode) {
      try{synthetic(device,width,mode,dtype);++cases;}
      catch(...){std::cerr<<"state VJP case width="<<width<<" mode="<<mode<<" reference="<<dtype<<'\n';throw;}
    }
    for(auto dtype:references)for(Index width:{1,257})for(int mode:{1,2,3})
    for(double rho:{0.,1.,-.5,.99123})for(Index chunk:{2,32})for(Index period:{1,3}) {
      try{synthetic(device,width,mode,dtype,true,rho,chunk,period);++cases;}
      catch(...){std::cerr<<"Add VJP width="<<width<<" mode="<<mode<<" reference="<<dtype<<" rho="<<rho
        <<" chunk="<<chunk<<" period="<<period<<'\n';throw;}
    }
    for(bool prefill:{false,true})for(bool repeat:{false,true})actual_tape(device,prefill,repeat,args.dtype);
    if(args.dtype==at::kHalf)for(Index chunk:{17,1024})rounding_sensitive(device,chunk);
    std::cout<<"device-state-vjp: passed isolated="<<cases<<" actual_device_tapes=4 rounding_anchors="<<(args.dtype==at::kHalf?2:0)
      <<" adjoints=FP32 scope=identity_EMA_Add_HARD_first_order_not_graph_training\n";
    runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
