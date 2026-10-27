// 0770593 allocation §4.2–4.4/S.3: real accelerator rows, branch maps and replay.
#include "weighted_sending.h"
#include "tide/weighted_sending.h"
#include "portable_torch/runtime.hpp"
#include <torch/csrc/autograd/autograd.h>
#include <ATen/Parallel.h>
#include <cmath>
#include <iostream>
namespace {
using namespace tide::device_online;
using namespace tide::device_online::weighted_sending;
void require(bool b,const char* text){if(!b)throw std::runtime_error(text);}
void close(const at::Tensor& a,const at::Tensor& b){
  require(a.sizes()==b.sizes()&&a.scalar_type()==b.scalar_type(),"sending comparison metadata");
  if(!at::allclose(a.cpu(),b,1e-5,1e-6)){
    std::cerr<<"actual="<<a.cpu()<<" expected="<<b<<'\n';throw std::runtime_error("resident sending numeric difference");}
}
struct Fixture {
  at::Tensor scores,caps,received,parameters,ids,lengths,root,cotangent;
  explicit Fixture(int variant){
    scores=at::tensor({.4f,-.7f,1.1f,-.2f,0.f}).repeat({6,1});scores+=variant*.07f;
    caps=at::tensor({.25f,.5f,0.f,.75f,0.f}).repeat({6,1});caps[1].zero_();received=caps*.5;
    parameters=at::tensor({.3f,.7f}).repeat({6,1});parameters[1][0]=0;parameters[3][0]=.2;parameters[5][0]=.2;
    ids=at::tensor(std::vector<int64_t>{(1LL<<60)+9,(1LL<<60)+2,(1LL<<60)+4,(1LL<<60)+1,0},at::kLong).repeat({6,1});
    lengths=at::tensor(variant?std::vector<int64_t>{3,4,0,1,2,1}:std::vector<int64_t>{4,3,0,2,3,1},at::kLong);
    root=at::tensor({1,1,1,1,0,1},at::kInt).to(at::kBool);
    cotangent=at::tensor({.3f,-.7f,.9f,.5f,.1f}).repeat({6,1});scores[3].fill_(-1);
    for(int64_t row=0;row<6;++row){auto n=lengths[row].item<int64_t>();
      scores[row].slice(0,n).fill_(NAN);caps[row].slice(0,n).fill_(-13);received[row].slice(0,n).fill_(NAN);
      cotangent[row].slice(0,n).fill_(NAN);ids[row].slice(0,n).fill_(-1);}
  }
};
void compare(const Fixture& f,const WeightedSendingTape& tape,const WeightedSendingGradient& gradients){
  auto values=tape.controls.cpu(),dv=gradients.vectors.cpu(),ds=gradients.scalars.cpu(),flags=gradients.connected.cpu();
  at::AutoGradMode mode(true);
  for(int64_t row=0;row<6;++row){
    auto n=f.lengths[row].item<int64_t>();
    if(!n){require(!flags[row].any().item<bool>()&&values[row].eq(0).all().item<bool>(),"absent sending row");continue;}
    std::vector<at::Tensor> xs{f.scores[row].slice(0,0,n),f.caps[row].slice(0,0,n),f.received[row].slice(0,0,n)};
    for(auto x:f.parameters[row].unbind())xs.push_back(x);
    for(auto& x:xs)x=x.clone().set_requires_grad(true);
    std::vector<int64_t> ids;for(int64_t j=0;j<n;++j)ids.push_back(f.ids[row][j].item<int64_t>());
    at::Tensor expected;
    if(tape.kind<=Pooled)expected=tide::weighted::attention_send(xs[0],xs[1],xs[2],
      tape.kind==Direct?"direct_attention":tape.kind==Unit?"unit_attention":"pooled_attention",
      tape.base==CapBase?"mu":tape.base==ReceivedBase?"received":"uniform",xs[4]);
    else if(tape.kind==Proportional)expected=tide::weighted::proportional(xs[1],tape.k);
    else if(tape.kind==SortedFill)expected=tide::weighted::total_zero(xs[0],xs[1],xs[3],ids);
    else expected=tide::weighted::hard_selection(xs[0],xs[1],tape.k,ids);
    try{
      close(values[row].slice(0,0,n),expected.detach());std::vector<at::Tensor> wanted(5);
      if(f.root[row].item<bool>())wanted=torch::autograd::grad({expected},xs,{f.cotangent[row].slice(0,0,n)},false,false,true);
      std::vector<at::Tensor> actual;for(auto x:dv[row].unbind())actual.push_back(x.slice(0,0,n));
      for(auto x:ds[row].unbind())actual.push_back(x);
      for(int64_t j=0;j<5;++j){
        require(flags[row][j].item<bool>()==wanted[j].defined(),"sending None/zero differs");
        if(wanted[j].defined())close(actual[j],wanted[j]);else require(actual[j].eq(0).all().item<bool>(),"absent sending gradient value");}
      require(values[row].slice(0,n).eq(0).all().item<bool>()&&dv[row].slice(1,n).eq(0).all().item<bool>(),"sending padding entered results");
    }catch(...){std::cerr<<"sending comparison kind="<<int(tape.kind)<<" base="<<int(tape.base)<<" row="<<row<<'\n';throw;}
  }
}
void check(at::Device device,Kind kind,Base base,int64_t k){
  at::NoGradGuard guard;Fixture first(0);
  auto s=first.scores.to(device),mu=first.caps.to(device),w=first.received.to(device),b=first.parameters.to(device);
  auto ids=first.ids.to(device),n=first.lengths.to(device),roots=first.root.to(device),g=first.cotangent.to(device);
  auto error=at::zeros({1},n.options().dtype(at::kInt));DeviceProgram p(device);p.limit_workspace(1024*1024);
  auto tape=append_weighted_sending(p,s,mu,w,b,ids,n,error,kind,base,k);
  auto grads=append_weighted_sending_vjp(p,tape,g,roots,error);
  auto committed=at::full_like(s,-7),ok=at::zeros({1},roots.options()),index=at::zeros_like(error),zero=at::zeros_like(error);
  auto publish=p.label(),done=p.label();p.equal(error,zero,ok);p.cast_index(ok,index);
  p.branch(index,{done,publish});p.mark(publish);p.copy(committed,tape.controls);p.mark(done);p.finish();
  for(int variant:{0,1}){
    Fixture input(variant);s.copy_(input.scores);mu.copy_(input.caps);w.copy_(input.received);b.copy_(input.parameters);
    ids.copy_(input.ids);n.copy_(input.lengths);roots.copy_(input.root);g.copy_(input.cotangent);error.zero_();
    portable_torch::synchronize(device);p.run();
    require(error.cpu().item<int>()==0,"resident sending refused valid frame");compare(input,tape,grads);close(committed,tape.controls.cpu());
  }
  auto before=committed.cpu();mu[0][0].fill_(2);portable_torch::synchronize(device);p.run();
  require(error.cpu().item<int>()==20,"resident sending accepted invalid cap");close(committed,before);p.run();close(committed,before);p.close();
  DeviceProgram refusal(device);bool rejected=false;
  try{append_weighted_sending(refusal,s,mu,w,b,ids,n,error,kind,base,k,1);}catch(const std::invalid_argument&){rejected=true;}
  require(rejected,"resident sending omitted scratch budget admission");refusal.close();
}
}
int main(int argc,char** argv){
  portable_torch::RuntimeSession runtime;
  try{
    auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("weighted sending requires explicit accelerator/FP32 scoring");
    auto device=portable_torch::resolve_device(args);at::set_num_threads(1);at::set_num_interop_threads(1);
    check(device,Direct,CapBase,0);check(device,Direct,ReceivedBase,0);check(device,Unit,CapBase,0);
    check(device,Unit,UniformBase,0);check(device,Pooled,CapBase,0);check(device,Proportional,CapBase,1);
    check(device,SortedFill,CapBase,0);check(device,Hard,CapBase,2);
    std::cout<<"weighted-sending: passed configs=8 replay_rows=96 logical_frames=80 complete_vjp=true host_decisions_per_send=0\n";return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
