#include "weighted_reverse_route.h"
#include "portable_torch/runtime.hpp"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <iostream>
#include <limits>
namespace {
using namespace tide::device_online;
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void same(const at::Tensor& x,const at::Tensor& y){require(at::equal(x.cpu(),y),"message ancestry adjoint differs from hand calculation");}
void run(at::Device device){
  at::NoGradGuard ng;constexpr int64_t huge=INT64_C(18014398509481987);
  auto coords=at::tensor(std::vector<int64_t>{0,1,6,1,0,4,0,1,6,1,1,4,0,0,9,1,2,8,1,0,huge,0,0,0,0,1,4,1,0,1,1,1,9,1,2,8},at::kLong).reshape({6,6});
  auto options=at::TensorOptions().device(device).dtype(at::kFloat);
  AtomBatch atoms{coords.to(device),at::zeros({6,3},options),at::ones({6},options.dtype(at::kBool))};
  auto indices=at::arange(6,at::kLong).reshape({2,3}).to(device),lengths=at::full({2},3,options.dtype(at::kLong));
  auto data=at::tensor({1.f,2.f,3.f,-1.f,4.f,0.f,.5f,0.f,1.f,0.f,0.f,0.f,2.f,-1.f,.5f,NAN,NAN,NAN},at::kFloat).reshape({2,3,3});
  auto gradient=data.to(device),roots=at::tensor(std::vector<int64_t>{1,1,1,1,1,0},at::kLong).to(device,at::kBool).reshape({2,3});
  auto edges=at::tensor(std::vector<int64_t>{0,0,1},at::kLong).to(device);
  auto times=at::zeros({3,2,2},at::kLong),present=at::zeros({3,2,2},at::kBool);
  times[0][0][0]=4;present[0][0][0]=true;times[1][1][0]=8;present[1][1][0]=true;times[1][1][1]=8;present[1][1][1]=true;
  WeightedReverseEvents e{times.to(device),present.to(device),at::zeros({3,2,2,3},options),at::zeros({3,2,2},options.dtype(at::kBool))};
  auto boundary=at::stack({coords[4],coords[3]});WeightedReverseBoundary b{boundary.to(device),at::ones({2},options.dtype(at::kBool)),at::zeros({2,3},options),at::zeros({2},options.dtype(at::kBool))};
  auto stage=at::full({1},2,options.dtype(at::kLong)),cut=at::full({1},3,options.dtype(at::kLong)),error=at::zeros({1},options.dtype(at::kInt));
  DeviceProgram p(device);append_weighted_reverse_route(p,atoms,indices,lengths,gradient,roots,edges,e,b,stage,cut,error);p.finish();
  for(int test=0;test<12;++test){
    atoms.coordinates.copy_(coords);atoms.valid.fill_(true);gradient.copy_(data);e.times.copy_(times);e.present.copy_(present);
    e.values.fill_(NAN);e.connected.zero_();b.coordinates.copy_(boundary);b.present.fill_(true);b.values.fill_(NAN);b.connected.zero_();stage.fill_(2);error.zero_();
    if(test==1){e.values[0][0][0].copy_(at::tensor({4.f,5.f,6.f}));e.connected[0][0][0]=true;}
    else if(test==2)e.present[0][0][0]=false;
    else if(test==3){e.present[1][0][0]=true;e.times[1][0][0]=4;}
    else if(test==4)stage.fill_(0);
    else if(test==5)b.present[0]=false;
    else if(test==6)b.coordinates[0].copy_(b.coordinates[1]);
    else if(test==7)atoms.coordinates[0][4]=3;
    else if(test==8)atoms.valid[0]=false;
    else if(test==9)gradient[0][0][0]=NAN;
    else if(test==10){
      atoms.coordinates.narrow(0,0,2).select(1,2).fill_(huge+2);atoms.coordinates.narrow(0,0,2).select(1,5).fill_(huge);
      e.times[0][0][0]=huge;e.present[1][0][0]=true;e.times[1][0][0]=huge-1;
    }else if(test==11){
      e.values[0][0][0][0]=std::numeric_limits<float>::max();e.connected[0][0][0]=true;
      gradient[0][0][0]=std::numeric_limits<float>::max();
    }
    portable_torch::synchronize(device);p.run();const auto code=error.cpu().item<int>();
    if(test>=2&&test!=10){require(code==((test==9||test==11)?37:41),"invalid reverse ancestry/root accepted");
      if(test==11){auto saved=e.values.cpu();gradient.zero_();portable_torch::synchronize(device);p.run();
        require(error.cpu().item<int>()==37,"reverse error was not sticky");
        require(at::equal(at::isnan(saved),at::isnan(e.values.cpu()))&&at::equal(at::nan_to_num(saved),at::nan_to_num(e.values.cpu())),"failed reverse result mutated on retry");}
      continue;}
    require(!code,"valid reverse ancestry rejected");
    same(e.values[0][0][0],test==1?at::tensor({4.f,11.f,9.f}):at::tensor({0.f,6.f,3.f}));same(e.values[1][1][0],at::tensor({.5f,0.f,1.f}));
    auto expected=at::zeros({3,2,2},at::kBool);expected[0][0][0]=expected[1][1][0]=true;same(e.connected,expected);
    same(b.values,at::tensor({2.f,-1.f,.5f,0.f,0.f,0.f}).reshape({2,3}));require(b.connected.cpu().all().item<bool>(),"zero boundary root disconnected");
    if(test!=10){same(atoms.coordinates,coords);same(e.times,times);}same(b.coordinates,boundary);
  }
  p.close();
  DeviceProgram malformed(device);bool denied=false;
  try{append_weighted_reverse_route(malformed,atoms,indices,lengths,gradient.to(at::kHalf),roots,edges,e,b,stage,cut,error);}
  catch(const std::invalid_argument&){denied=true;}require(denied,"implicit reverse dtype substitution accepted");malformed.close();
}
}
int main(int argc,char** argv){portable_torch::RuntimeSession runtime;
  try{auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("weighted reverse route requires explicit accelerator FP32");
    auto device=portable_torch::resolve_device(args);at::set_num_threads(1);at::set_num_interop_threads(1);run(device);
    std::cout<<"weighted-reverse-route: passed cases=12 actual_ancestry=true exact_time=true parallel_edges=true none_zero=true\n";return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
