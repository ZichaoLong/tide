#include "weighted_residual.h"
#include "portable_torch/runtime.hpp"
#include <torch/csrc/autograd/autograd.h>
#include <ATen/Parallel.h>
#include <iostream>
namespace {
using namespace tide::device_online;using at::Tensor;
constexpr int R=6,B=3,D=3,M=2,F=5;
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void close(const Tensor& a,const Tensor& b){require(a.sizes()==b.sizes(),"residual shape mismatch");
  if(!at::allclose(a.cpu(),b,1e-5,1e-6)){std::cerr<<a.cpu()<<" versus "<<b<<'\n';throw std::runtime_error("residual numerical mismatch");}}
Tensor values(at::IntArrayRef shape,float shift){int64_t n=1;for(auto x:shape)n*=x;return (at::arange(n,at::kFloat)*.13+shift).sin().reshape(shape)*.2;}
std::vector<Tensor> roles(const WeightedResidualParameters& w){return {w.out,w.gate,w.up,w.down};}
struct Formula {std::vector<Tensor> leaves,outputs;};
Formula reference(const Tensor& input,const Tensor& memory,const Tensor& owners,const WeightedResidualParameters& w,double am,double af){
  at::AutoGradMode enabled(true);Formula f;std::vector<std::vector<Tensor>> params(4);
  for(int r=0;r<R;++r)f.leaves.push_back(input[r].clone().set_requires_grad(true));
  for(int r=0;r<R;++r)f.leaves.push_back(memory[r].clone().set_requires_grad(true));
  for(int j=0;j<4;++j)for(int b=0;b<B;++b){auto x=roles(w)[j][b].clone().set_requires_grad(true);params[j].push_back(x);f.leaves.push_back(x);}
  for(int r=0;r<R;++r){const int bank=owners[r].item<int64_t>();if(bank<0){f.outputs.push_back(at::zeros({D},at::kFloat));continue;}
    auto x=f.leaves[r]+am*at::mv(params[0][bank],f.leaves[R+r]);auto normal=x/(x.square().mean()+.13).sqrt();
    auto gate=at::mv(params[1][bank],normal),up=at::mv(params[2][bank],normal);
    f.outputs.push_back(x+af*at::mv(params[3][bank],(gate*gate.sigmoid())*up));}
  return f;
}
void compare(const Formula& ref,const Tensor& bar,const Tensor& roots,const WeightedResidualTape& t,const WeightedResidualGradient& g){
  at::AutoGradMode enabled(true);std::vector<Tensor> outputs,bars;
  for(int r=0;r<R;++r){close(t.output[r],ref.outputs[r]);if(roots[r].item<bool>()&&ref.outputs[r].requires_grad()){outputs.push_back(ref.outputs[r]);bars.push_back(bar[r]);}}
  auto expected=torch::autograd::grad(outputs,ref.leaves,bars,true,false,true);auto flags=g.connected.cpu();
  for(int r=0;r<R;++r){require(flags[r].item<bool>()==expected[r].defined()&&flags[r].item<bool>()==expected[R+r].defined(),"residual private root differs");
    if(expected[r].defined()){close(g.input[r],expected[r]);close(g.memory[r],expected[R+r]);}
    else require(g.input[r].cpu().eq(0).all().item<bool>()&&g.memory[r].cpu().eq(0).all().item<bool>(),"absent residual gradients nonzero");}
  auto parameters=roles(g.parameters);std::vector<Tensor> connected{g.out_connected.cpu(),g.gate_connected.cpu(),g.up_connected.cpu(),g.down_connected.cpu()};
  for(int role=0;role<4;++role)for(int bank=0;bank<B;++bank){const auto& e=expected[2*R+role*B+bank];
    require(connected[role][bank].item<bool>()==e.defined(),"residual parameter None/zero differs");
    if(e.defined())close(parameters[role][bank],e);else require(parameters[role][bank].cpu().eq(0).all().item<bool>(),"absent residual bank gradient nonzero");}
}
void check(at::Device device,double am,double af){
  at::NoGradGuard ng;auto opt=at::TensorOptions().device(device).dtype(at::kFloat);
  auto input=at::zeros({R,D},opt),memory=at::zeros({R,M},opt),owners=at::zeros({R},opt.dtype(at::kLong));
  auto bar=at::zeros_like(input),roots=at::zeros({R},opt.dtype(at::kBool)),error=at::zeros({1},opt.dtype(at::kInt));
  WeightedResidualParameters w{at::zeros({B,D,M},opt),at::zeros({B,F,D},opt),at::zeros({B,F,D},opt),at::zeros({B,D,F},opt)};
  DeviceProgram p(device);p.limit_workspace(64*1024*1024);
  auto t=append_weighted_residual(p,input,memory,owners,w,.13,am,af,error);auto g=append_weighted_residual_vjp(p,t,bar,roots,error);p.finish();
  for(int pass=0;pass<2;++pass){auto u=values({R,D},.1+pass*.1),m=values({R,M},.2);u[4].fill_(NAN);u[5].fill_(NAN);m[0].zero_();m[4].fill_(NAN);m[5].fill_(NAN);
    auto ids=at::tensor(std::vector<int64_t>{0,1,pass?2:0,1,-1,-1},at::kLong),flags=at::tensor({1,1,pass,1,1,0},at::kInt).to(at::kBool);
    auto dy=values({R,D},.8);dy[3].zero_();dy[4].fill_(NAN);dy[5].fill_(NAN);
    WeightedResidualParameters weights{values({B,D,M},.3+pass*.1),values({B,F,D},.5),values({B,F,D},.7),values({B,D,F},.9)};
    if(!pass)for(auto x:roles(weights))x[2].fill_(NAN);
    auto ref=reference(u,m,ids,weights,am,af);input.copy_(u);memory.copy_(m);owners.copy_(ids);bar.copy_(dy);roots.copy_(flags);
    auto host=roles(weights),target=roles(w);for(int j=0;j<4;++j)target[j].copy_(host[j]);
    portable_torch::synchronize(device);p.run();require(!error.cpu().item<int>(),"residual rejected valid actual rows");compare(ref,dy,flags,t,g);
    if(pass){DeviceProgram reverse(device);auto saved=append_weighted_residual_vjp(reverse,t,bar,roots,error);reverse.finish();
      input.fill_(NAN);memory.fill_(NAN);owners.fill_(-99);for(auto x:roles(w))x.fill_(NAN);
      portable_torch::synchronize(device);reverse.run();require(!error.cpu().item<int>(),"residual reverse reread original values");compare(ref,dy,flags,t,saved);reverse.close();}
  }
  portable_torch::synchronize(device);p.run();require(error.cpu().item<int>()!=0,"residual invalid input accepted");p.close();
  DeviceProgram denied(device);bool refused=false;try{append_weighted_residual(denied,input,memory,owners,w,.13,am,af,error,1);}catch(const std::invalid_argument&){refused=true;}
  require(refused,"residual scratch budget omitted");denied.close();
}
}
int main(int argc,char** argv){portable_torch::RuntimeSession runtime;
  try{auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("residual requires explicit accelerator/FP32");
    auto device=portable_torch::resolve_device(args);at::set_num_threads(1);at::set_num_interop_threads(1);
    for(double am:{0.,.3})for(double af:{0.,.7})check(device,am,af);
    std::cout<<"weighted-residual: passed configurations=4 replay_rows=48 shared_banks=3 complete_vjp=true connected_zero_scales=true saved_versions=true\n";return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
