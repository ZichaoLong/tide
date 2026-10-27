#include "weighted_projection.h"
#include "portable_torch/runtime.hpp"
#include <torch/csrc/autograd/autograd.h>
#include <ATen/Parallel.h>
#include <iostream>
namespace {
using namespace tide::device_online;using at::Tensor;
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void close(const Tensor& a,const Tensor& b){
  require(a.sizes()==b.sizes()&&a.scalar_type()==b.scalar_type(),"projection comparison layout");
  if(!at::allclose(a.cpu(),b,1e-5,1e-6)){std::cerr<<a.cpu()<<" versus "<<b<<'\n';throw std::runtime_error("weighted projection numeric difference");}
}
Tensor values(at::IntArrayRef shape,float shift){int64_t n=1;for(auto x:shape)n*=x;return (at::arange(n,at::kFloat)*.13+shift).sin().reshape(shape)*.2;}
struct Formula {std::vector<Tensor> inputs,matrix,bias,output;};
Formula reference(const Tensor& x,const Tensor& ids,const WeightedProjectionParameters& w,bool norm){
  at::AutoGradMode enabled(true);Formula ref;
  for(int bank=0;bank<3;++bank){ref.matrix.push_back(w.matrix[bank].clone().set_requires_grad(true));
    if(w.bias.defined())ref.bias.push_back(w.bias[bank].clone().set_requires_grad(true));}
  for(int row=0;row<6;++row){auto input=x[row].clone().set_requires_grad(true);ref.inputs.push_back(input);
    const auto bank=ids[row].item<int64_t>();if(bank<0){ref.output.push_back(at::zeros({4},at::kFloat));continue;}
    auto normalized=norm?input/(input.square().mean()+.13).sqrt():input;
    auto y=at::mv(ref.matrix[bank],normalized);if(w.bias.defined())y=y+ref.bias[bank];ref.output.push_back(y);}
  return ref;
}
void compare(const Formula& ref,const Tensor& ids,const Tensor& bar,const Tensor& flags,
    const WeightedProjectionTape& t,const WeightedProjectionGradient& g){
  at::AutoGradMode enabled(true);std::vector<Tensor> roots,bars,leaves=ref.inputs;
  leaves.insert(leaves.end(),ref.matrix.begin(),ref.matrix.end());leaves.insert(leaves.end(),ref.bias.begin(),ref.bias.end());
  for(int row=0;row<6;++row){close(t.output[row],ref.output[row].to(t.output.scalar_type()));if(ids[row].item<int64_t>()>=0&&flags[row].item<bool>()){roots.push_back(ref.output[row]);bars.push_back(bar[row]);}}
  const auto expected=torch::autograd::grad(roots,leaves,bars,true,false,true);auto input=g.input.cpu(),matrix=g.parameters.matrix.cpu(),row_flags=g.row_connected.cpu(),owner_flags=g.owner_connected.cpu();
  for(int row=0;row<6;++row){require(row_flags[row].item<bool>()==expected[row].defined(),"projection input None/zero differs");
    if(expected[row].defined())close(input[row],expected[row]);else require(input[row].eq(0).all().item<bool>(),"absent projection input gradient nonzero");}
  for(int bank=0;bank<3;++bank){require(owner_flags[bank].item<bool>()==expected[6+bank].defined(),"projection matrix None/zero differs");
    if(expected[6+bank].defined())close(matrix[bank],expected[6+bank]);
    if(!ref.bias.empty()){require(owner_flags[bank].item<bool>()==expected[9+bank].defined(),"projection bias root differs");if(expected[9+bank].defined())close(g.parameters.bias[bank],expected[9+bank]);}}
  require(g.parameters.bias.defined()==!ref.bias.empty(),"projection fabricated bias role");
}
void check(at::Device device,bool norm,bool bias,at::ScalarType dtype){
  at::NoGradGuard no_grad;auto opt=at::TensorOptions().device(device).dtype(dtype);
  auto x=at::zeros({6,3},opt),owners=at::zeros({6},opt.dtype(at::kLong)),bar=at::zeros({6,4},opt.dtype(at::kFloat)),flags=at::zeros({6},opt.dtype(at::kBool));
  auto error=at::zeros({1},opt.dtype(at::kInt));WeightedProjectionParameters w{at::zeros({3,4,3},opt),bias?at::zeros({3,4},opt):Tensor{}};
  DeviceProgram p(device);p.limit_workspace(64*1024*1024);std::optional<double> epsilon=norm?std::optional<double>(.13):std::nullopt;
  auto t=append_weighted_projection(p,x,owners,w,epsilon,error);auto g=append_weighted_projection_vjp(p,t,bar,flags,error);p.finish();
  for(int pass=0;pass<2;++pass){auto input=values({6,3},.1+pass*.1).to(dtype).to(at::kFloat);input[0].zero_();input[4].fill_(NAN);input[5].fill_(NAN);
    auto ids=at::tensor(std::vector<int64_t>{0,1,pass?2:0,1,-1,-1},at::kLong),roots=at::tensor({1,1,pass,1,1,0},at::kInt).to(at::kBool);
    auto dy=values({6,4},.8);dy[3].zero_();dy[4].fill_(NAN);dy[5].fill_(NAN);
    WeightedProjectionParameters weights{values({3,4,3},.3+pass*.1).to(dtype).to(at::kFloat),bias?values({3,4},.7).to(dtype).to(at::kFloat):Tensor{}};
    auto ref=reference(input,ids,weights,norm);x.copy_(input);owners.copy_(ids);bar.copy_(dy);flags.copy_(roots);w.matrix.copy_(weights.matrix);if(bias)w.bias.copy_(weights.bias);
    portable_torch::synchronize(device);p.run();require(!error.cpu().item<int>(),"projection refused valid rows");compare(ref,ids,dy,roots,t,g);
    if(pass){DeviceProgram reverse(device);auto saved=append_weighted_projection_vjp(reverse,t,bar,flags,error);reverse.finish();
      x.fill_(NAN);owners.fill_(-99);w.matrix.fill_(NAN);if(bias)w.bias.fill_(NAN);
      portable_torch::synchronize(device);reverse.run();require(!error.cpu().item<int>(),"projection saved reverse read changed inputs");compare(ref,ids,dy,roots,t,saved);reverse.close();}
  }
  portable_torch::synchronize(device);p.run();require(error.cpu().item<int>()!=0,"projection accepted invalid owners/values");p.close();
  DeviceProgram refusal(device);bool rejected=false;
  try{append_weighted_projection(refusal,x,owners,w,epsilon,error,1);}catch(const std::invalid_argument&){rejected=true;}
  require(rejected,"projection scratch budget omitted");refusal.close();
}
void half_storage(at::Device device){
  at::NoGradGuard ng;auto opt=at::TensorOptions().device(device).dtype(at::kHalf);
  auto x=at::tensor({1.,.5,0.},at::kFloat).reshape({3,1}).to(device,at::kHalf);
  auto ids=at::zeros({3},opt.dtype(at::kLong)),roots=at::ones({3},opt.dtype(at::kBool)),error=at::zeros({1},opt.dtype(at::kInt));
  WeightedProjectionParameters w{at::full({1,1,1},0x1p-24,opt),{}};
  auto bar=at::ones({3,1},opt.dtype(at::kFloat));DeviceProgram p(device);
  auto t=append_weighted_projection(p,x,ids,w,std::nullopt,error);auto g=append_weighted_projection_vjp(p,t,bar,roots,error);p.finish();
  portable_torch::synchronize(device);p.run();require(!error.cpu().item<int>(),"half projection rejected subnormal");
  require(at::equal(t.output.cpu(),at::tensor({0x1p-24,0.,0.},at::kFloat).reshape({3,1}).to(at::kHalf)),"half projection subnormal/underflow differs");
  require(at::equal(g.input.cpu(),at::full({3,1},0x1p-24,at::kFloat)),"half projection lost subnormal adjoint");
  close(g.parameters.matrix,at::full({1,1,1},1.5,at::kFloat));
  require(g.row_connected.cpu().all().item<bool>()&&g.owner_connected.cpu().all().item<bool>(),"half underflow changed projection roots");
  x.fill_(65504);w.matrix.fill_(2);portable_torch::synchronize(device);p.run();
  require(error.cpu().item<int>()!=0,"finite FP32 projection overflowed half silently");p.close();
  DeviceProgram invalid(device);bool refused=false;
  try{append_weighted_projection_vjp(invalid,t,bar.to(at::kHalf),roots,error);}catch(const std::invalid_argument&){refused=true;}
  require(refused,"half projection accepted non-FP32 cotangent");invalid.close();
}
}
int main(int argc,char** argv){portable_torch::RuntimeSession runtime;
  try{auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||(args.dtype!=at::kFloat&&args.dtype!=at::kHalf))throw std::invalid_argument("weighted projection requires explicit accelerator/FP32 or FP16");
    auto device=portable_torch::resolve_device(args);at::set_num_threads(1);at::set_num_interop_threads(1);
    for(bool norm:{false,true})for(bool bias:{false,true})check(device,norm,bias,args.dtype);
    if(args.dtype==at::kHalf)half_storage(device);
    std::cout<<"weighted-projection: passed configurations=4 replay_rows=48 shared_banks=3 all_vjps=true none_zero=true saved_versions=true adjoint_dtype=float32\n";return 0;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}}
