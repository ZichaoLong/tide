// 0770593 allocation §6/S.2: actual accelerator forward and exact first-order map.
#include "weighted_allocation.h"
#include "tide/weighted.h"
#include "portable_torch/runtime.hpp"
#include <torch/csrc/autograd/autograd.h>
#include <ATen/Parallel.h>
#include <cmath>
#include <iostream>
namespace {
using namespace tide::device_online;
void require(bool ok,const char* text){if(!ok)throw std::runtime_error(text);}
void close(const at::Tensor& a,const at::Tensor& b){
  require(a.sizes()==b.sizes()&&a.scalar_type()==b.scalar_type(),"allocation comparison metadata");
  if(!at::allclose(a.cpu(),b,1e-5,1e-6)){
    std::cerr<<"actual="<<a.cpu()<<" expected="<<b<<'\n';
    throw std::runtime_error("resident allocation numeric difference");
  }
}
struct Fixture {
  at::Tensor scores,caps,bounds,lengths,root,cotangent;
  explicit Fixture(int variant){
    scores=at::tensor({.4f,-.7f,1.1f,-.2f,0.f}).repeat({6,1});
    scores+=variant*.07f;
    caps=at::tensor({.25f,.5f,.125f,.75f,0.f}).repeat({6,1});
    bounds=at::tensor({.2f,.6f,.8f,.1f}).repeat({6,1});
    lengths=at::tensor(variant?std::vector<int64_t>{3,4,0,1,2,1}:std::vector<int64_t>{4,3,0,2,3,1},at::kLong);
    root=at::tensor({1,0,1,1,1,1},at::kInt).to(at::kBool);
    cotangent=at::tensor({.3f,-.7f,.9f,.5f,.1f}).repeat({6,1});
    bounds[1][0]=.6;bounds[1][1]=.6;bounds[3][0]=0;bounds[3][1]=0;
    bounds[4][0]=caps[4].narrow(0,0,lengths[4].item<int64_t>()).sum();bounds[4][1]=1;
    bounds[5][0]=.2;bounds[5][1]=.2;
    for(int64_t i=0;i<6;++i){auto n=lengths[i].item<int64_t>();
      scores[i].slice(0,n).fill_(NAN);caps[i].slice(0,n).fill_(-13);cotangent[i].slice(0,n).fill_(NAN);}
  }
};
void compare(const Fixture& f,const WeightedAllocationTape& tape,const WeightedAllocationGradient& gradients,
             weighted_allocation::Rule rule,float alpha){
  auto values=tape.controls.cpu(),branch=tape.branches.cpu(),v=gradients.vectors.cpu();
  auto scalars=gradients.scalars.cpu(),flags=gradients.connected.cpu();
  const std::vector<std::string> names{"zero","caps","slack","lower","upper","equality"};
  at::AutoGradMode grad_mode(true);
  for(int64_t row=0;row<6;++row){
    std::cerr<<"compare rule="<<int(rule)<<" alpha="<<alpha<<" row="<<row<<'\n';
    auto n=f.lengths[row].item<int64_t>();
    if(!n){require(branch[row].item<int64_t>()==-1&&!flags[row].any().item<bool>(),"absent allocation row");continue;}
    std::vector<at::Tensor> xs{f.scores[row].slice(0,0,n),f.caps[row].slice(0,0,n)};
    for(auto scalar:f.bounds[row].unbind())xs.push_back(scalar);
    for(auto& x:xs)x=x.clone().set_requires_grad(true);
    auto expected=tide::weighted::allocate(xs[0],xs[1],xs[2],xs[3],xs[4],
      rule==weighted_allocation::Quad?"quad":rule==weighted_allocation::Bernoulli?"bern":"ent",
      rule==weighted_allocation::Entropy?std::optional<double>(alpha):std::nullopt,xs[5]);
    require(names.at(branch[row].item<int64_t>())==expected.constraint,"allocation branch differs");
    close(values[row].slice(0,0,n),expected.controls.detach());
    std::vector<at::Tensor> wanted(6);
    if(f.root[row].item<bool>()&&expected.controls.requires_grad())
      wanted=torch::autograd::grad({expected.controls},xs,{f.cotangent[row].slice(0,0,n)},false,false,true);
    std::vector<at::Tensor> actual{v[row][0].slice(0,0,n),v[row][1].slice(0,0,n)};
    for(auto scalar:scalars[row].unbind())actual.push_back(scalar);
    for(int64_t j=0;j<6;++j){
      require(flags[row][j].item<bool>()==wanted[j].defined(),"allocation None/zero differs");
      if(wanted[j].defined()){
        std::cerr<<"gradient field="<<j<<'\n';close(actual[j],wanted[j]);
      }else require(actual[j].eq(0).all().item<bool>(),"absent allocation gradient value");
    }
    require(values[row].slice(0,n).eq(0).all().item<bool>(),"allocation padding entered output");
  }
}
void check(at::Device device,weighted_allocation::Rule rule,float alpha){
  at::NoGradGuard guard;Fixture first(0);
  auto s=first.scores.to(device),mu=first.caps.to(device),b=first.bounds.to(device),n=first.lengths.to(device);
  auto root=first.root.to(device),g=first.cotangent.to(device);
  auto error=at::zeros({1},n.options().dtype(at::kInt));
  DeviceProgram p(device);p.limit_workspace(1024*1024);
  auto tape=append_weighted_allocation(p,s,mu,b,n,error,rule,alpha);
  auto grads=append_weighted_allocation_vjp(p,tape,g,root,error);
  auto committed=at::full_like(s,-7),ok=at::zeros({1},root.options()),index=at::zeros_like(error),zero=at::zeros_like(error);
  auto publish=p.label(),done=p.label();p.equal(error,zero,ok);p.cast_index(ok,index);
  p.branch(index,{done,publish});p.mark(publish);p.copy(committed,tape.controls);p.mark(done);p.finish();
  for(int variant:{0,1}){
    Fixture input(variant);s.copy_(input.scores);mu.copy_(input.caps);b.copy_(input.bounds);
    n.copy_(input.lengths);root.copy_(input.root);g.copy_(input.cotangent);error.zero_();
    portable_torch::synchronize(device);p.run();
    std::cerr<<"run rule="<<int(rule)<<" alpha="<<alpha<<" variant="<<variant<<" error="<<error.cpu()<<'\n';
    require(error.cpu().item<int>()==0,"resident allocation refused valid frame");
    compare(input,tape,grads,rule,alpha);close(committed,tape.controls.cpu());
  }
  auto before=committed.cpu();mu[0][0].fill_(2);portable_torch::synchronize(device);p.run();
  require(error.cpu().item<int>()==weighted_allocation::Domain,"resident allocation accepted invalid cap");
  close(committed,before);p.run();close(committed,before);p.close();
  DeviceProgram refusal(device);bool rejected=false;
  try{append_weighted_allocation(refusal,s,mu,b,n,error,rule,alpha,true,1);}catch(const std::invalid_argument&){rejected=true;}
  require(rejected,"resident allocation omitted explicit scratch admission");refusal.close();
}
}
int main(int argc,char** argv){
  portable_torch::RuntimeSession runtime;
  try{
    auto args=portable_torch::parse_cli(argc,argv,true);
    if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("weighted allocation requires explicit accelerator and FP32 scoring");
    auto device=portable_torch::resolve_device(args);at::set_num_threads(1);at::set_num_interop_threads(1);
    check(device,weighted_allocation::Quad,1);check(device,weighted_allocation::Bernoulli,1);
    for(float alpha:{1.f,1.000001f,1.5f,2.f})check(device,weighted_allocation::Entropy,alpha);
    std::cout<<"weighted-allocation: passed rules=6 replay_rows=72 logical_frames=60 complete_vjp=true host_decisions_per_solve=0\n";return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
