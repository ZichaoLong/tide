#pragma once
#include "parameter_vjp.h"
#include "portable_torch/runtime.hpp"
#include <ATen/core/grad_mode.h>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace tide::device_online::test {
inline void private_accumulation_check(at::Device device) {
  at::NoGradGuard guard;
  auto require=[](bool yes,const char* message){if(!yes)throw std::runtime_error(message);};
  auto exact=[&](const at::Tensor& a,const at::Tensor& b,const char* message) {
    require(at::equal(a.cpu().view(at::kByte),b.cpu().view(at::kByte)),message);
  };
  int cases=0;
  // More than 32 tiles exposes block-zero flag publication racing readers
  // when a previously absent owner becomes connected. Tail sizes span tiles.
  for(int64_t width:{1,63,255,256,257,8193,1048583}) {
    ParameterRegistry registry;
    for(int i=0;i<5;++i)registry.add("owner"+std::to_string(i),at::zeros({width},at::kFloat));
    const auto owners=registry.owners();const std::vector<int64_t> offsets{0,width,2*width,3*width,-1};
    for(int round=0;round<4;++round) {
      auto av=at::arange(4*width,at::kFloat).remainder(17)*.125f;
      auto bv=at::arange(4*width,at::kFloat).remainder(13)*-.0625f;
      std::vector<uint8_t> ac(5),bc(5);
      for(int i=0;i<4;++i){ac[i]=((i+round)%4)>=2;bc[i]=((i+round)%2)!=0;}
      auto expected=at::zeros_like(av);
      for(int i=0;i<4;++i) {
        auto out=expected.narrow(0,i*width,width);
        if(ac[i])out.copy_(av.narrow(0,i*width,width));
        if(bc[i])out.add_(bv.narrow(0,i*width,width));
        if(!ac[i])av.narrow(0,i*width,width).fill_(std::numeric_limits<float>::quiet_NaN());
        if(!bc[i])bv.narrow(0,i*width,width).fill_(std::numeric_limits<float>::quiet_NaN());
      }
      auto a_flags=at::tensor(ac,at::kByte).to(at::kBool),b_flags=at::tensor(bc,at::kByte).to(at::kBool);
      ParameterVjp a{owners,offsets,av.to(device),a_flags.to(device)},b{owners,offsets,bv.to(device),b_flags.to(device)};
      auto error=at::zeros({1},a.values.options().dtype(at::kInt));
      CannProgram program(device);auto out=append_private_parameter_accumulate(program,a,b,error,128*1024*1024);
      require(out.values.is_alias_of(a.values)&&!out.connected.is_alias_of(a.connected)
        &&!out.connected.is_alias_of(b.connected),"private accumulation storage roles changed");
      program.finish();portable_torch::synchronize(device);program.run();
      require(!error.cpu().item<int>(),"private accumulation refused valid owner layout");
      require(at::equal(out.values.cpu(),expected)&&at::equal(out.connected.cpu(),a_flags|b_flags),
        "private accumulation changed order/None/poison/tail semantics");
      exact(a.connected,a_flags,"private accumulation mutated old connection flags");
      exact(b.values,bv,"private accumulation mutated current backward values");
      exact(b.connected,b_flags,"private accumulation mutated current backward connections");
      program.close();++cases;
      if(round==0) {
        bool refused=false;
        try{CannProgram small(device);append_private_parameter_accumulate(small,a,b,error,1);}
        catch(const std::invalid_argument&){refused=true;}
        require(refused,"private accumulation bypassed admission");
        refused=false;
        try{CannProgram alias(device);append_private_parameter_accumulate(alias,a,a,error,128*1024*1024);}
        catch(const std::invalid_argument&){refused=true;}
        require(refused,"private accumulation accepted overlapping sources");
        auto before=a.values.clone();error.fill_(7);
        CannProgram failed(device);append_private_parameter_accumulate(failed,a,b,error,128*1024*1024);failed.finish();
        portable_torch::synchronize(device);failed.run();exact(a.values,before,"upstream refusal changed private values");failed.close();
      }
    }
  }
  auto f=at::TensorOptions().device(device).dtype(at::kFloat);
  ParameterVjp empty{{},{},at::ones({1},f),at::zeros({1},f.dtype(at::kBool))};
  auto other=empty;other.values=at::zeros_like(empty.values);other.connected=empty.connected.clone();
  auto error=at::zeros({1},f.dtype(at::kInt));CannProgram program(device);
  auto out=append_private_parameter_accumulate(program,empty,other,error,1024);program.finish();
  portable_torch::synchronize(device);program.run();
  require(!error.cpu().item<int>()&&!out.values.cpu().item<float>()&&!out.connected.cpu().item<bool>(),
    "empty private accumulation fabricated an owner");program.close();
  std::cout<<"private-accumulation: passed numeric_cases="<<cases<<" admission_alias_error_cases=21 empty=1 flags_separate=true\n";
}
} // namespace tide::device_online::test
