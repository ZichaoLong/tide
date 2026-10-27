// 0770593 allocation §3/S.1: actual batched affine/SiLU + complete shared VJP.
#include "weighted_head.h"
#include "portable_torch/runtime.hpp"
#include <torch/csrc/autograd/autograd.h>
#include <ATen/Parallel.h>
#include <cmath>
#include <iostream>
namespace {
using namespace tide::device_online;
void require(bool value,const char* text){if(!value)throw std::runtime_error(text);}
void close(const at::Tensor& x,const at::Tensor& y){
  require(x.sizes()==y.sizes()&&x.scalar_type()==y.scalar_type(),"head comparison metadata");
  if(!at::allclose(x.cpu(),y,1e-5,1e-6)){
    std::cerr<<"actual="<<x.cpu()<<" expected="<<y<<'\n';throw std::runtime_error("resident head numeric difference");}
}
std::vector<at::Tensor> tensors(const WeightedHeadParameters& p){return {p.matrix,p.hidden_bias,p.projection,p.bias};}
at::Tensor values(at::IntArrayRef shape,float shift){
  int64_t count=1;for(auto d:shape)count*=d;
  return (at::arange(count,at::kFloat)*.11+shift).sin().reshape(shape)*.2;
}
struct Fixture {
  at::Tensor input,owners,gradient,roots;
  WeightedHeadParameters parameters;
  Fixture(int64_t in,int64_t hidden,int64_t out,int variant):
    input(values({6,in},variant*.1)),gradient(values({6,out},.7)),
    parameters{values({3,hidden,in},.1+variant*.01),values({3,hidden},.2),values({3,out,hidden},.3),values({3,out},.4)}{
    owners=at::tensor(variant?std::vector<int64_t>{1,-1,0,1,0,-1}:std::vector<int64_t>{0,1,0,-1,1,-1},at::kLong);
    roots=at::tensor({1,0,1,1,0,1},at::kInt).to(at::kBool);if(!variant)gradient[0].zero_();
    for(int64_t i=0;i<6;++i){if(owners[i].item<int64_t>()<0)input[i].fill_(NAN);
      if(!roots[i].item<bool>()||owners[i].item<int64_t>()<0)gradient[i].fill_(NAN);}
  }
};
void compare(const Fixture& f,const WeightedHeadTape& tape,const WeightedHeadGradient& actual){
  at::AutoGradMode mode(true);std::vector<at::Tensor> xs,parameters,outputs,gradients,rows;
  for(auto x:f.input.unbind())xs.push_back(x.clone().set_requires_grad(true));
  for(auto bank:tensors(f.parameters))for(auto x:bank.unbind())parameters.push_back(x.clone().set_requires_grad(true));
  for(int64_t row=0;row<6;++row){
    const auto owner=f.owners[row].item<int64_t>();
    if(owner<0){rows.push_back(at::zeros_like(f.gradient[row]));continue;}
    auto h=at::silu(at::matmul(parameters[owner],xs[row])+parameters[3+owner]);
    auto y=at::matmul(parameters[6+owner],h)+parameters[9+owner];rows.push_back(y.detach());
    if(f.roots[row].item<bool>()){outputs.push_back(y);gradients.push_back(f.gradient[row]);}
  }
  close(tape.output,at::stack(rows));std::vector<at::Tensor> inputs=xs;inputs.insert(inputs.end(),parameters.begin(),parameters.end());
  auto expected=torch::autograd::grad(outputs,inputs,gradients,false,false,true);
  auto flags=actual.row_connected.cpu(),owner_flags=actual.owner_connected.cpu();
  for(int64_t row=0;row<6;++row){
    require(flags[row].item<bool>()==expected[row].defined(),"head row None/zero differs");
    if(expected[row].defined())close(actual.input[row],expected[row]);else require(actual.input[row].cpu().eq(0).all().item<bool>(),"absent head input gradient");}
  auto banks=tensors(actual.parameters);
  for(int64_t bank=0;bank<3;++bank)for(int64_t field=0;field<4;++field){
    const auto& target=expected[6+field*3+bank];
    require(owner_flags[bank].item<bool>()==target.defined(),"head shared owner None/zero differs");
    if(target.defined())close(banks[field][bank],target);else require(banks[field][bank].cpu().eq(0).all().item<bool>(),"absent head parameter gradient");}
}
void check(at::Device device,int64_t in,int64_t hidden,int64_t out){
  at::NoGradGuard no_grad;Fixture first(in,hidden,out,0);
  auto x=first.input.to(device),owners=first.owners.to(device),g=first.gradient.to(device),roots=first.roots.to(device);
  auto move=[&](at::Tensor v){return v.to(device);};
  WeightedHeadParameters w{move(first.parameters.matrix),move(first.parameters.hidden_bias),move(first.parameters.projection),move(first.parameters.bias)};
  auto error=at::zeros({1},owners.options().dtype(at::kInt));DeviceProgram p(device);p.limit_workspace(64*1024*1024);
  auto tape=append_weighted_head(p,x,owners,w,error);auto gradient=append_weighted_head_vjp(p,tape,g,roots,error);
  auto committed=at::full_like(tape.output,-7),ok=at::zeros({1},roots.options()),index=at::zeros_like(error),zero=at::zeros_like(error);
  auto publish=p.label(),done=p.label();p.equal(error,zero,ok);p.cast_index(ok,index);
  p.branch(index,{done,publish});p.mark(publish);p.copy(committed,tape.output);p.mark(done);p.finish();
  std::cout<<"head-workspace: dimensions="<<in<<','<<hidden<<','<<out<<" forward_and_vjp="<<p.workspace_bytes()<<'\n';
  for(int variant:{0,1}){
    Fixture f(in,hidden,out,variant);x.copy_(f.input);owners.copy_(f.owners);g.copy_(f.gradient);roots.copy_(f.roots);
    auto sources=tensors(f.parameters),destinations=tensors(w);for(int j=0;j<4;++j)destinations[j].copy_(sources[j]);
    portable_torch::synchronize(device);p.run();require(error.cpu().item<int>()==0,"resident head refused valid rows");
    compare(f,tape,gradient);close(committed,tape.output.cpu());
  }
  // Reverse consumes actual saved input/owner/weight versions, not mutable banks.
  DeviceProgram reverse(device);reverse.limit_workspace(64*1024*1024);
  auto again=append_weighted_head_vjp(reverse,tape,g,roots,error);reverse.finish();
  x.fill_(NAN);owners.fill_(42);for(auto v:tensors(w))v.fill_(NAN);
  portable_torch::synchronize(device);reverse.run();require(error.cpu().item<int>()==0,"head reverse reread mutable banks");
  compare(Fixture(in,hidden,out,1),tape,again);reverse.close();
  // Bad owner metadata cannot reach out-of-range vendor gathers or publication.
  auto before=committed.cpu();p.run();require(error.cpu().item<int>()==2,"resident head accepted invalid owner");close(committed,before);p.close();
  DeviceProgram refusal(device);bool refused=false;
  try{append_weighted_head(refusal,x,owners,w,error,1);}catch(const std::invalid_argument&){refused=true;}
  require(refused,"resident head omitted scratch admission");refusal.close();
}
}
int main(int argc,char** argv){
  portable_torch::RuntimeSession runtime;
  try{
    auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("resident head requires explicit accelerator/FP32");
    auto device=portable_torch::resolve_device(args);at::set_num_threads(1);at::set_num_interop_threads(1);
    check(device,4,5,1);check(device,3,3,3);
    std::cout<<"weighted-head: passed profiles=2 rows=24 complete_vjp=true shared_parameters=true saved_versions=true vendor_batched=true\n";return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
