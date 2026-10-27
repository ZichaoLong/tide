// Complete fixed-time cache/current/shared-parameter VJP and raw-update composition.
#include "weighted_convolution.h"
#include "portable_torch/runtime.hpp"
#include <torch/csrc/autograd/autograd.h>
#include <ATen/Parallel.h>
#include <iostream>
#include <limits>
namespace {
using namespace tide;using namespace tide::device_online;
constexpr int64_t big=INT64_C(18014398509481987),last=std::numeric_limits<int64_t>::max();
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void close(const Tensor& a,const Tensor& b){
  require(a.sizes()==b.sizes()&&a.scalar_type()==b.scalar_type(),"convolution comparison metadata");
  if(!at::allclose(a.cpu(),b,1e-6,1e-7)){std::cerr<<a.cpu()<<" versus "<<b<<'\n';throw std::runtime_error("fixed-time convolution differs");}
}
Tensor values(at::IntArrayRef shape,double shift){int64_t n=1;for(auto x:shape)n*=x;
  return (at::arange(n,at::kFloat)*.13+shift).sin().mul(.2).reshape(shape);}
std::vector<State> fixture(const WeightedRawLayout& l,at::ScalarType dtype=at::kFloat){
  std::vector<std::vector<int64_t>> stamps{{},{big,big+1,big+3},{0,1,3},{last-5,last-2},{0}};
  std::vector<State> out;
  for(size_t r=0;r<stamps.size();++r){auto& ts=stamps[r];while(int64_t(ts.size())>l.horizon()+1)ts.erase(ts.begin());
    State s;s.last_time=ts.empty()?-1:ts.back();s.observations=ts.empty()?0:8;
    for(const auto& f:l.fields()){auto shape=f.shape;if(f.ragged)shape.insert(shape.begin(),ts.size());
      auto x=values(shape,.3+r).to(dtype);if(f.name=="value")s.value=at::zeros_like(x);else s.slots[f.name]=f.name=="W"?x.abs():x;}
    s.slots["time"]=at::full({},std::max<int64_t>(0,s.last_time),at::kLong);s.slots["conv_times"]=at::tensor(ts,at::kLong);out.push_back(s);
  }return out;
}
struct Formula {std::vector<Tensor> output,cache,current,weight,bias;};
Formula reference(const std::vector<State>& states,const Tensor& current,const Tensor& time,const Tensor& owners,
    const WeightedConvParameters& w,int64_t stride){
  at::AutoGradMode grad(true);Formula ref;const auto size=w.weight.size(1),features=current.size(1);
  for(int64_t bank=0;bank<w.weight.size(0);++bank){ref.weight.push_back(w.weight[bank].to(at::kFloat).clone().set_requires_grad(true));ref.bias.push_back(w.bias[bank].to(at::kFloat).clone().set_requires_grad(true));}
  for(size_t row=0;row<states.size();++row){auto z=current[row].to(at::kFloat).clone().set_requires_grad(true),cache=states[row].slots.at("conv_values").to(at::kFloat).clone().set_requires_grad(true);
    ref.cache.push_back(cache);ref.current.push_back(z);const auto bank=owners[row].item<int64_t>();if(bank<0){ref.output.push_back(at::zeros_like(z));continue;}
    // Independent dense original-time signal and grouped conv1d. The current
    // entry in a proposal cache is deliberately excluded from this signal.
    auto y=ref.bias[bank]+ref.weight[bank][0]*z;const auto n=cache.size(0),theta=time[row].item<int64_t>();
    if(size>1&&n){const auto horizon=(size-1)*stride;auto match=at::zeros({horizon,n},at::kFloat);
      for(int64_t i=0;i<horizon;++i)for(int64_t j=0;j<n;++j)
        if(theta>=horizon-i&&theta-(horizon-i)==states[row].slots.at("conv_times")[j].item<int64_t>())match[i][j]=1;
      auto sequence=at::cat({at::matmul(match,cache),z.unsqueeze(0)}).t().unsqueeze(0);
      auto kernel=ref.weight[bank].flip({0}).t().unsqueeze(1);
      y=at::conv1d(sequence,kernel,ref.bias[bank],at::IntArrayRef{1},at::IntArrayRef{0},at::IntArrayRef{stride},features).flatten();}
    ref.output.push_back(y);
  }return ref;
}
void compare(const WeightedRawLayout& l,const Formula& ref,const Tensor& bar,const Tensor& roots,const Tensor& owners,
    const WeightedConvTape& t,const WeightedConvGradient& g){
  at::AutoGradMode enable(true);std::vector<Tensor> leaves=ref.cache,ys,bars;
  leaves.insert(leaves.end(),ref.current.begin(),ref.current.end());leaves.insert(leaves.end(),ref.weight.begin(),ref.weight.end());leaves.insert(leaves.end(),ref.bias.begin(),ref.bias.end());
  for(size_t r=0;r<ref.output.size();++r){close(t.output[r],ref.output[r].to(t.output.scalar_type()));if(roots[r].item<bool>()&&owners[r].item<int64_t>()>=0){ys.push_back(ref.output[r]);bars.push_back(bar[r]);}}
  const auto expected=torch::autograd::grad(ys,leaves,bars,true,false,true);auto gc=g.raw_cache.cpu(),gz=g.current.cpu(),gw=g.parameters.weight.cpu(),gb=g.parameters.bias.cpu();
  auto cf=g.cache_connected.cpu(),zf=g.row_connected.cpu(),wf=g.owner_connected.cpu();const auto n=ref.cache.size(),banks=ref.weight.size();
  for(size_t r=0;r<n;++r){require(cf[r].item<bool>()==expected[r].defined()&&zf[r].item<bool>()==expected[n+r].defined(),"convolution private roots differ");
    if(expected[r].defined())close(gc[r].slice(0,l.field("conv_values").offset,l.field("conv_values").offset+expected[r].numel()).reshape(expected[r].sizes()),expected[r]);
    if(expected[n+r].defined())close(gz[r],expected[n+r]);}
  for(size_t b=0;b<banks;++b){require(wf[b].item<bool>()==expected[2*n+b].defined()&&wf[b].item<bool>()==expected[2*n+banks+b].defined(),"convolution shared owner roots differ");
    if(expected[2*n+b].defined()){close(gw[b],expected[2*n+b]);close(gb[b],expected[2*n+banks+b]);}}
}
void run(at::Device device,int64_t size,at::ScalarType dtype){
  at::NoGradGuard no_grad;weighted::RecurrentConfig config;config.memory_family="gated_delta";config.width=3;config.heads=2;
  config.key_width=2;config.value_width=2;config.conv_size=size;config.conv_stride=2;WeightedRawLayout l(config,5);
  auto states=fixture(l,dtype);auto cpu=encode_weighted_raw_state(l,states),old=copy_weighted_raw_state(cpu,device);const auto features=config.conv_width();
  auto opt=old.floating.options();auto current=at::zeros({5,features},opt),time=at::zeros({5},opt.dtype(at::kLong)),owners=at::zeros({5},opt.dtype(at::kLong));
  auto bar=at::zeros(current.sizes(),opt.dtype(at::kFloat)),roots=at::zeros({5},opt.dtype(at::kBool)),error=at::zeros({1},opt.dtype(at::kInt));
  WeightedConvParameters weights{at::zeros({3,size,features},opt),at::zeros({3,features},opt)};
  DeviceProgram p(device);p.limit_workspace(64*1024*1024);auto t=append_weighted_convolution(p,l,old,time,current,owners,weights,2,error);
  auto g=append_weighted_convolution_vjp(p,t,bar,roots,error);auto owner=copy_weighted_raw_state(cpu,device);
  append_weighted_raw_adopt(p,owner,old,error);p.finish();
  for(int pass=0;pass<2;++pass){
    auto z=values({5,features},.2+pass*.1).to(dtype),dy=values({5,features},.8);z[4].fill_(NAN);dy[3].zero_();dy[4].fill_(NAN);
    auto theta=at::tensor(std::vector<int64_t>{0,big+4,8,last,-1},at::kLong),ids=at::tensor(std::vector<int64_t>{0,1,pass?1:0,1,-1},at::kLong);
    auto flags=at::tensor({1,1,pass,1,1},at::kInt).to(at::kBool);WeightedConvParameters w{values({3,size,features},.3+pass*.3).to(dtype),values({3,features},.7).to(dtype)};
    if(pass&&states[1].slots.at("conv_times").numel()>1){states[1].slots["conv_times"]=states[1].slots["conv_times"].slice(0,1).clone();states[1].slots["conv_values"]=states[1].slots["conv_values"].slice(0,1).clone();}
    cpu=encode_weighted_raw_state(l,states);old.floating.copy_(cpu.floating);old.integer.copy_(cpu.integer);
    current.copy_(z);time.copy_(theta);owners.copy_(ids);bar.copy_(dy);roots.copy_(flags);weights.weight.copy_(w.weight);weights.bias.copy_(w.bias);
    auto ref=reference(states,z,theta,ids,w,2);portable_torch::synchronize(device);p.run();require(!error.cpu().item<int>(),"fixed-time convolution valid device row refused");compare(l,ref,dy,flags,ids,t,g);
    if(pass){DeviceProgram reverse(device);auto saved=append_weighted_convolution_vjp(reverse,t,bar,roots,error);reverse.finish();
      old.integer.fill_(-99);old.floating.fill_(NAN);weights.weight.fill_(NAN);weights.bias.fill_(NAN);current.fill_(NAN);owners.fill_(-99);time.fill_(-1);
      portable_torch::synchronize(device);reverse.run();require(!error.cpu().item<int>(),"fixed-time saved reverse read overwritten sources");compare(l,ref,dy,flags,ids,t,saved);reverse.close();}
  }
  auto before=copy_weighted_raw_state(owner,at::kCPU);portable_torch::synchronize(device);p.run();require(error.cpu().item<int>()==27,"invalid convolution owner accepted");
  require(at::equal(owner.floating.cpu(),before.floating)&&at::equal(owner.integer.cpu(),before.integer),"failed convolution committed owner");p.close();
  DeviceProgram tiny(device);bool refused=false;try{append_weighted_convolution(tiny,l,old,time,current,owners,weights,2,error,1);}catch(const std::invalid_argument&){refused=true;}
  require(refused,"convolution tensor budget omitted");tiny.close();
}
void snapshots(at::Device device,at::ScalarType dtype){
  at::NoGradGuard no_grad;weighted::RecurrentConfig config;config.memory_family="ssd";config.width=3;config.key_width=2;config.value_width=2;config.conv_size=3;config.conv_stride=2;
  WeightedRawLayout l(config,5);auto states=fixture(l,dtype);states.resize(1);auto raw=encode_weighted_raw_state(l,states),old=copy_weighted_raw_state(raw,device);
  const auto features=config.conv_width();auto z=values({1,features},.3).to(dtype),input=z.to(device),error=at::zeros({1},old.integer.options().dtype(at::kInt));
  auto theta=at::zeros({1},old.integer.options()),id=at::zeros_like(theta),mass=at::full({1},.25,old.floating.options()),yes=at::ones({1},old.integer.options().dtype(at::kBool));
  auto proposed=old.floating.clone();proposed[0].slice(0,l.field("conv_values").offset,l.field("conv_values").offset+features).copy_(input[0]);
  WeightedConvParameters w{values({1,3,features},.4).to(dtype).to(device),values({1,features},.6).to(dtype).to(device)};
  DeviceProgram p(device);auto next=append_weighted_raw_update(p,l,old,theta,mass,yes,proposed,error);
  auto a=append_weighted_convolution(p,l,old,theta,input,id,w,2,error),b=append_weighted_convolution(p,l,next.result,theta,input,id,w,2,error);
  auto bg=append_weighted_convolution_vjp(p,b,at::ones(input.sizes(),input.options().dtype(at::kFloat)),yes,error);auto field_roots=at::zeros({1,int64_t(l.fields().size())},yes.options());
  // Registered copy consumes the actual device connection after convolution.
  p.copy(field_roots.select(1,l.fields().size()-1),bg.cache_connected);
  auto chain=append_weighted_raw_update_vjp(p,next,bg.raw_cache,field_roots,error);p.finish();portable_torch::synchronize(device);p.run();
  require(!error.cpu().item<int>(),"raw-update/convolution composition refused");close(a.output,b.output.cpu());
  require(chain.mass_connected.cpu().item<bool>()&&chain.mass.cpu().eq(0).all().item<bool>(),"current slot fabricated a mass dependence or lost connected zero");
  p.close();
}
void absent_tap_overflow(at::Device device){
  at::NoGradGuard no_grad;weighted::RecurrentConfig c;c.memory_family="gated_delta";c.width=3;c.key_width=1;c.value_width=1;c.conv_size=2;c.conv_stride=2;
  WeightedRawLayout l(c,3);auto states=fixture(l);states.resize(1);auto old=copy_weighted_raw_state(encode_weighted_raw_state(l,states),device);
  auto f=old.floating.options();auto time=at::zeros({1},old.integer.options()),id=at::zeros_like(time),z=at::zeros({1,c.conv_width()},f);
  WeightedConvParameters w{at::zeros({1,2,c.conv_width()},f),at::zeros_like(z)};w.weight[0][1].fill_(1e30);
  auto error=at::zeros({1},f.dtype(at::kInt)),roots=at::ones({1},f.dtype(at::kBool));DeviceProgram p(device);
  auto t=append_weighted_convolution(p,l,old,time,z,id,w,2,error);auto g=append_weighted_convolution_vjp(p,t,at::full_like(z,1e20),roots,error);p.finish();
  portable_torch::synchronize(device);p.run();require(!error.cpu().item<int>(),"missing historical tap overflow fabricated invalid returned gradient");
  require(!g.cache_connected.cpu().item<bool>()&&g.owner_connected.cpu().item<bool>()&&g.current.cpu().eq(0).all().item<bool>()
    &&g.parameters.weight.cpu().eq(0).all().item<bool>(),"missing tap gradient support changed");close(g.parameters.bias,at::full({1,c.conv_width()},1e20,at::kFloat));p.close();
}
void half_storage(at::Device device){
  at::NoGradGuard ng;weighted::RecurrentConfig c;c.memory_family="ssd";c.width=3;c.key_width=1;c.value_width=1;c.conv_size=1;
  WeightedRawLayout l(c,1);auto states=fixture(l,at::kHalf);states.resize(1);
  auto raw=encode_weighted_raw_state(l,states),old=copy_weighted_raw_state(raw,device),owner=copy_weighted_raw_state(raw,device);
  auto f=old.floating.options();auto time=at::zeros({1},old.integer.options()),ids=at::zeros_like(time),x=at::ones({1,c.conv_width()},f);
  auto roots=at::ones({1},f.dtype(at::kBool)),error=at::zeros({1},f.dtype(at::kInt)),bar=at::ones(x.sizes(),f.dtype(at::kFloat));
  WeightedConvParameters w{at::full({1,1,c.conv_width()},0x1p-24,f),at::zeros_like(x)};
  DeviceProgram p(device);auto t=append_weighted_convolution(p,l,old,time,x,ids,w,1,error);
  auto g=append_weighted_convolution_vjp(p,t,bar,roots,error);append_weighted_raw_adopt(p,owner,old,error);p.finish();
  for(double scale:{1.,.5}){x.fill_(scale);portable_torch::synchronize(device);p.run();
    require(!error.cpu().item<int>(),"half convolution rejected subnormal/underflow");
    require(at::equal(t.output.cpu(),at::full(x.sizes(),scale==1.?0x1p-24:0.,at::kHalf)),"half convolution subnormal/underflow differs");
    require(at::equal(g.current.cpu(),at::full(x.sizes(),0x1p-24,at::kFloat)),"half convolution lost subnormal adjoint");
    require(g.row_connected.cpu().item<bool>()&&g.owner_connected.cpu().item<bool>(),"half underflow changed convolution roots");}
  auto before=copy_weighted_raw_state(owner,at::kCPU);old.floating.fill_(.3);x.fill_(65504);w.weight.fill_(2);
  portable_torch::synchronize(device);p.run();require(error.cpu().item<int>()==27,"finite FP32 convolution overflowed half silently");
  require(at::equal(owner.floating.cpu(),before.floating)&&at::equal(owner.integer.cpu(),before.integer),"half overflow committed raw owner");p.close();
  DeviceProgram invalid(device);bool refused=false;
  try{append_weighted_convolution_vjp(invalid,t,bar.to(at::kHalf),roots,error);}catch(const std::invalid_argument&){refused=true;}
  require(refused,"half convolution accepted non-FP32 cotangent");invalid.close();
}
}
int main(int argc,char** argv){portable_torch::RuntimeSession runtime;
  try{auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||(args.dtype!=at::kFloat&&args.dtype!=at::kHalf))throw std::invalid_argument("weighted convolution requires explicit accelerator FP32/FP16");
    auto device=portable_torch::resolve_device(args);at::set_num_threads(1);at::set_num_interop_threads(1);run(device,1,args.dtype);run(device,3,args.dtype);snapshots(device,args.dtype);
    if(args.dtype==at::kFloat)absent_tap_overflow(device);else half_storage(device);
    std::cout<<"weighted-convolution: passed tap_counts=2 replay_rows=20 shared_banks=3 saved_reverse=true raw_update_composition=true int64_taps=true failed_no_commit=true adjoint_dtype=float32\n";return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
