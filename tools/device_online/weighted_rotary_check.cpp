#include "weighted_rotary.h"
#include "portable_torch/runtime.hpp"
#include <torch/csrc/autograd/autograd.h>
#include <ATen/Parallel.h>
#include <iostream>
namespace {
using namespace tide::device_online;using at::Tensor;
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void close(const Tensor& a,const Tensor& b){require(a.sizes()==b.sizes(),"rotation result shape mismatch");if(!at::allclose(a.cpu(),b,2e-5,2e-6)){
  std::cerr<<a.cpu()<<" versus "<<b<<'\n';throw std::runtime_error("relative rotation numeric mismatch");}}
Tensor values(float shift){return (at::arange(30,at::kFloat)*.13+shift).sin().reshape({2,3,5});}
std::pair<Tensor,Tensor> reference(const Tensor& input,const Tensor& ages,const std::vector<double>& f,const Tensor& bar){
  at::AutoGradMode enabled(true);auto x=input.clone().set_requires_grad(true);std::vector<Tensor> rows;
  for(int r=0;r<2;++r){std::vector<Tensor> entries;for(int i=0;i<3;++i){auto matrix=at::eye(5,at::kFloat);
    for(int64_t j=0;j<int64_t(f.size());++j){const auto a=5-2*int64_t(f.size())+2*j;
      auto phase=-ages[r][i].to(at::kFloat)/7*float(f[j]),c=phase.cos(),s=phase.sin();
      matrix[a][a]=c;matrix[a][a+1]=-s;matrix[a+1][a]=s;matrix[a+1][a+1]=c;}
    entries.push_back(at::mv(matrix,x[r][i]));}rows.push_back(at::stack(entries));}
  auto output=at::stack(rows),dx=torch::autograd::grad({output},{x},{bar})[0];return {output,dx};
}
void check(at::Device device,int pairs){
  at::NoGradGuard ng;auto opt=at::TensorOptions().device(device).dtype(at::kFloat);
  auto x=at::zeros({2,3,5},opt),ages=at::zeros({2,3},opt.dtype(at::kLong)),bar=at::zeros_like(x),roots=at::zeros({2},opt.dtype(at::kBool)),error=at::zeros({1},opt.dtype(at::kInt));
  std::vector<double> f{.13,-.07};f.resize(pairs);DeviceProgram p(device);p.limit_workspace(64*1024*1024);
  auto t=append_weighted_rotary(p,x,ages,7,f,error);auto dx=append_weighted_rotary_vjp(p,t,bar,roots,error);p.finish();
  for(int pass=0;pass<2;++pass){auto input=values(.2+pass*.1),dy=values(.7);auto times=at::tensor(std::vector<int64_t>{0,1,3,INT64_C(16777217),INT64_C(18014398509481987),INT64_MAX},at::kLong).reshape({2,3});
    if(pass)times[0][2]=4;auto expected=reference(input,times,f,dy);x.copy_(input);ages.copy_(times);bar.copy_(dy);roots.fill_(true);
    portable_torch::synchronize(device);p.run();require(!error.cpu().item<int>(),"rotation refused valid ages");
    if(pairs){auto phase=-times.to(at::kFloat).unsqueeze(-1)/7*at::tensor(f,at::kFloat);require(at::equal(t.phase.cpu(),phase),"rotation int64 to FP32 phase differs");}
    close(t.output,expected.first);close(dx,expected.second);
    if(pass){DeviceProgram reverse(device);auto saved=append_weighted_rotary_vjp(reverse,t,bar,roots,error);reverse.finish();x.fill_(NAN);ages.fill_(-99);roots[1]=false;bar[1].fill_(NAN);
      portable_torch::synchronize(device);reverse.run();require(!error.cpu().item<int>(),"rotation reverse reread original source");close(saved[0],expected.second[0]);require(saved[1].cpu().eq(0).all().item<bool>(),"absent rotation bar read");reverse.close();}
  }
  portable_torch::synchronize(device);p.run();require(error.cpu().item<int>()==33,"invalid rotation source accepted");p.close();
  DeviceProgram denied(device);bool refused=false;try{append_weighted_rotary(denied,x,ages,7,f,error,1);}catch(const std::invalid_argument&){refused=true;}
  require(refused,"rotation scratch capacity omitted");denied.close();
}
}
int main(int argc,char** argv){portable_torch::RuntimeSession runtime;
  try{auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("weighted rotary needs explicit accelerator/FP32");
    auto device=portable_torch::resolve_device(args);at::set_num_threads(1);at::set_num_interop_threads(1);
    for(int pairs:{0,1,2})check(device,pairs);
    std::cout<<"weighted-rotary: passed configurations=3 replay_rows=12 partial_rotation=true extreme_ages=true saved_vjp=true\n";return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
