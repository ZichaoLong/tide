#include "weighted_projection.h"
#include "weighted_storage_budget.h"
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
  for(int bank=0;bank<w.matrix.size(0);++bank){ref.matrix.push_back(w.matrix[bank].clone().set_requires_grad(true));
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
  std::vector<Tensor> expected(leaves.size());if(!roots.empty())expected=torch::autograd::grad(roots,leaves,bars,true,false,true);
  auto input=g.input.cpu(),matrix=g.parameters.matrix.cpu(),row_flags=g.row_connected.cpu(),owner_flags=g.owner_connected.cpu();
  for(int row=0;row<6;++row){require(row_flags[row].item<bool>()==expected[row].defined(),"projection input None/zero differs");
    if(expected[row].defined())close(input[row],expected[row]);else require(input[row].eq(0).all().item<bool>(),"absent projection input gradient nonzero");}
  for(int bank=0;bank<ref.matrix.size();++bank){require(owner_flags[bank].item<bool>()==expected[6+bank].defined(),"projection matrix None/zero differs");
    if(expected[6+bank].defined())close(matrix[bank],expected[6+bank]);else require(matrix[bank].eq(0).all().item<bool>(),"absent projection matrix gradient nonzero");
    if(!ref.bias.empty()){const auto& e=expected[6+ref.matrix.size()+bank];require(owner_flags[bank].item<bool>()==e.defined(),"projection bias root differs");if(e.defined())close(g.parameters.bias[bank],e);}}
  require(g.parameters.bias.defined()==!ref.bias.empty(),"projection fabricated bias role");
}
void check(at::Device device,bool norm,bool bias,at::ScalarType dtype,int banks){
  at::NoGradGuard no_grad;auto opt=at::TensorOptions().device(device).dtype(dtype);
  auto x=at::zeros({6,3},opt),owners=at::zeros({6},opt.dtype(at::kLong)),bar=at::zeros({6,4},opt.dtype(at::kFloat)),flags=at::zeros({6},opt.dtype(at::kBool));
  auto error=at::zeros({1},opt.dtype(at::kInt));WeightedProjectionParameters w{at::zeros({banks,4,3},opt),bias?at::zeros({banks,4},opt):Tensor{}};
  DeviceProgram p(device);p.limit_workspace(64*1024*1024);std::optional<double> epsilon=norm?std::optional<double>(.13):std::nullopt;
  auto t=append_weighted_projection(p,x,owners,w,epsilon,error);auto g=append_weighted_projection_vjp(p,t,bar,flags,error);p.finish();
  require(t.matrix.size(0)==banks,"projection must save one actual matrix per parameter bank");
  for(int pass=0;pass<3;++pass){auto input=values({6,3},.1+pass*.1).to(dtype).to(at::kFloat);input[0].zero_();input[4].fill_(NAN);input[5].fill_(NAN);
    auto ids=at::tensor(std::vector<int64_t>{0,1,pass?2:0,1,-1,-1},at::kLong),roots=at::tensor({1,1,pass,1,1,0},at::kInt).to(at::kBool);
    auto dy=values({6,4},.8);dy[3].zero_();dy[4].fill_(NAN);dy[5].fill_(NAN);
    if(banks==1)ids.clamp_max_(0);
    WeightedProjectionParameters weights{values({banks,4,3},.3+pass*.1).to(dtype).to(at::kFloat),bias?values({banks,4},.7).to(dtype).to(at::kFloat):Tensor{}};
    if(pass==2){input.fill_(NAN);ids.fill_(-1);weights.matrix.fill_(NAN);if(bias)weights.bias.fill_(NAN);dy.fill_(NAN);}
    auto ref=reference(input,ids,weights,norm);x.copy_(input);owners.copy_(ids);bar.copy_(dy);flags.copy_(roots);w.matrix.copy_(weights.matrix);if(bias)w.bias.copy_(weights.bias);
    portable_torch::synchronize(device);p.run();require(!error.cpu().item<int>(),"projection refused valid rows");compare(ref,ids,dy,roots,t,g);
    if(pass==1){DeviceProgram reverse(device);auto saved=append_weighted_projection_vjp(reverse,t,bar,flags,error);reverse.finish();
      x.fill_(NAN);owners.fill_(-99);w.matrix.fill_(NAN);if(bias)w.bias.fill_(NAN);
      portable_torch::synchronize(device);reverse.run();require(!error.cpu().item<int>(),"projection saved reverse read changed inputs");compare(ref,ids,dy,roots,t,saved);reverse.close();}
  }
  owners.zero_();x.zero_();w.matrix.fill_(NAN);portable_torch::synchronize(device);p.run();
  require(error.cpu().item<int>()!=0,"projection accepted a used nonfinite bank");error.zero_();
  owners.fill_(-99);portable_torch::synchronize(device);p.run();require(error.cpu().item<int>()!=0,"projection accepted invalid owners/values");p.close();
  DeviceProgram refusal(device);bool rejected=false;
  try{append_weighted_projection(refusal,x,owners,w,epsilon,error,1);}catch(const std::invalid_argument&){rejected=true;}
  require(rejected,"projection scratch budget omitted");refusal.close();
}
void single_bank_capacity(at::Device device){
  at::NoGradGuard ng;constexpr int64_t rows=64,width=32,limit=65536;
  auto x=values({rows,width},.1).to(device),m=values({1,width,width},.3).to(device);
  auto ids=at::zeros({rows},x.options().dtype(at::kLong)),error=at::zeros({1},ids.options().dtype(at::kInt));
  auto bar=values({rows,width},.7).to(device),roots=at::ones({rows},ids.options().dtype(at::kBool));
  DeviceProgram p(device);p.limit_workspace(64*1024*1024);
  const WeightedStorageIncrement forward(p,{x,m,ids,error});
  auto t=append_weighted_projection(p,x,ids,{m,{}},std::nullopt,error,limit);const auto forward_bytes=forward.bytes();
  const WeightedStorageIncrement reverse(p,{t.input,t.rstd,t.owners,t.indices,t.matrix,t.bias,t.output,bar,roots,error});
  auto g=append_weighted_projection_vjp(p,t,bar,roots,error,limit);const auto reverse_bytes=reverse.bytes();p.finish();
  portable_torch::synchronize(device);p.run();require(!error.cpu().item<int>(),"single-bank capacity case refused");
  close(t.output,at::mm(x.cpu(),m.cpu()[0].t()));close(g.input,at::mm(bar.cpu(),m.cpu()[0]));
  close(g.parameters.matrix,at::mm(bar.cpu().t(),x.cpu()).unsqueeze(0));
  require(forward_bytes<=limit&&reverse_bytes<=limit&&2*rows*width*width*4>limit,"single-bank storage witness failed");
  require(g.row_connected.cpu().all().item<bool>()&&g.owner_connected.cpu().all().item<bool>(),"single-bank capacity roots differ");
  std::cout<<"single-bank projection bytes: forward="<<forward_bytes<<" reverse="<<reverse_bytes<<" allowance="<<limit
    <<" old_replicated_matrices="<<2*rows*width*width*4<<'\n';p.close();
  DeviceProgram refusal(device);bool refused=false;
  try{append_weighted_projection(refusal,x,ids,{m,{}},std::nullopt,error,forward_bytes-1);}catch(const std::invalid_argument&){refused=true;}
  require(refused,"single-bank actual storage floor omitted");refusal.close();
}
void multiple_bank_capacity(at::Device device,int64_t rows,bool bias){
  at::NoGradGuard ng;constexpr int64_t width=32,banks=3,limit=131072;
  auto x=values({rows,width},.13).to(device),m=values({banks,width,width},.37).to(device);
  auto bv=bias?values({banks,width},.23).to(device):Tensor{};
  auto ids=at::remainder(at::arange(rows,at::kLong),banks).to(device),error=at::zeros({1},ids.options().dtype(at::kInt));
  auto bar=values({rows,width},.71).to(device),roots=at::ones({rows},ids.options().dtype(at::kBool));
  DeviceProgram p(device);p.limit_workspace(64*1024*1024);
  const WeightedStorageIncrement f(p,{x,m,bv,ids,error});
  auto t=append_weighted_projection(p,x,ids,{m,bv},std::nullopt,error,limit);const auto forward=f.bytes();
  const WeightedStorageIncrement b(p,{t.input,t.rstd,t.owners,t.indices,t.matrix,t.bias,t.output,bar,roots,error});
  auto g=append_weighted_projection_vjp(p,t,bar,roots,error,limit);const auto reverse=b.bytes();p.finish();
  portable_torch::synchronize(device);p.run();require(!error.cpu().item<int>(),"multi-bank capacity case refused");
  auto xx=x.cpu(),mm=m.cpu(),gg=bar.cpu(),ii=ids.cpu();
  auto yy=at::zeros_like(xx),dx=at::zeros_like(xx),dw=at::zeros_like(mm);
  auto bb=bias?bv.cpu():Tensor{},db=at::zeros({banks,width},at::kFloat);
  // Independent per-row formula; the candidate never materializes these outers.
  for(int64_t row=0;row<rows;++row){const auto owner=ii[row].item<int64_t>();
    yy[row].copy_(at::mv(mm[owner],xx[row]));dx[row].copy_(at::mv(mm[owner].t(),gg[row]));
    dw[owner].add_(at::outer(gg[row],xx[row]));if(bias){yy[row].add_(bb[owner]);db[owner].add_(gg[row]);}}
  close(t.output,yy);close(g.input,dx);close(g.parameters.matrix,dw);
  if(bias)close(g.parameters.bias,db);
  require(t.matrix.size(0)==banks&&forward<=limit&&reverse<=limit,"multi-bank storage did not retain bank granularity");
  require(g.row_connected.cpu().all().item<bool>()&&g.owner_connected.cpu().all().item<bool>(),"multi-bank connections differ");
  std::cout<<"multi-bank projection bytes: rows="<<rows<<" bias="<<bias<<" forward="<<forward<<" reverse="<<reverse<<" allowance="<<limit
    <<" old_replicated_matrices="<<2*rows*width*width*4<<'\n';p.close();
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
    args.allow_npu_float16=true;auto device=portable_torch::resolve_device(args);at::set_num_threads(1);at::set_num_interop_threads(1);
    for(int banks:{1,3})for(bool norm:{false,true})for(bool bias:{false,true})check(device,norm,bias,args.dtype,banks);
    single_bank_capacity(device);multiple_bank_capacity(device,64,false);multiple_bank_capacity(device,193,true);
    if(args.dtype==at::kHalf)half_storage(device);
    std::cout<<"weighted-projection: passed configurations=8 single_bank=true banks3=true all_vjps=true none_zero=true saved_versions=true adjoint_dtype=float32\n";return 0;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}}
