#pragma once
#include "device_optimizer.h"
#include "optimizer_layout.h"
#include <portable_torch/runtime.hpp>
#include <ATen/core/grad_mode.h>
#include <iostream>
#include <limits>

namespace tide::device_online::test {
// Exercise the actual optimizer transaction, including comparison-word/tile
// tails, signed NaNs/infinities and disconnected poison. No CPU result feeds
// the candidate. Existing trajectories separately cover slots and proposals.
inline void optimizer_finite_check(at::Device device) {
  at::NoGradGuard guard;int probes=0;
  auto require=[](bool ok,const char* message){if(!ok)throw std::runtime_error(message);};
  for(int64_t width:{1,7,8,63,64,65,255,256,257}) {
    ParameterRegistry registry;auto parameter=at::zeros({width},at::kFloat);registry.add("weight",parameter);
    auto f=at::TensorOptions().device(device).dtype(at::kFloat);
    ParameterVjp gradient{registry.owners(),{0},at::zeros({width},f),at::ones({1},f.dtype(at::kBool))};
    OptimizerGroup group;group.parameters={"weight"};group.lr=.125;
    DeviceOptimizer optimizer(gradient,DeviceOptimizerKind::sgd,{group},1024*1024);
    auto error=at::zeros({1},f.dtype(at::kInt));CannProgram program(device);
    optimizer.append_step(program,gradient,error);program.finish();
    auto snapshot=[&] {std::vector<Tensor> out;for(const auto& x:{optimizer.values(),optimizer.first(),optimizer.second(),
      optimizer.maximum(),optimizer.steps(),optimizer.corrections()})out.push_back(x.cpu().view(at::kByte).clone());return out;};
    auto unchanged=[&](const std::vector<Tensor>& before){const auto after=snapshot();
      for(size_t i=0;i<after.size();++i)require(at::equal(before[i],after[i]),"finite rejection changed optimizer state");};
    // Full/tail comparisons must accept finite boundary cases, including
    // FLT_MAX and subnormals; an all-zero gradient leaves them finite.
    for(auto value:{0.f,-0.f,std::numeric_limits<float>::max(),-std::numeric_limits<float>::max(),
                   std::numeric_limits<float>::denorm_min(),-std::numeric_limits<float>::denorm_min()}) {
      optimizer.values().fill_(value);gradient.values.zero_();gradient.connected.fill_(true);error.zero_();
      portable_torch::synchronize(device);program.run();
      require(error.cpu().item<int>()==0,"finite optimizer value was rejected");++probes;
    }
    std::vector<int64_t> positions{0,width-1};if(width>64)positions.push_back(63);if(width>65)positions.push_back(64);
    for(auto bits:{uint32_t(0x7f800000),uint32_t(0xff800000),uint32_t(0x7fc00001),uint32_t(0xffc00001),
                   uint32_t(0x7f800001),uint32_t(0xff800001)})for(auto position:positions) {
      // Construct poison by its bits to cover both quiet and signalling NaNs.
      auto words=at::zeros({width},at::kInt);words[position].fill_(int64_t(int32_t(bits)));
      const auto poisoned=words.view(at::kFloat);
      require(!at::isfinite(poisoned[position]).item<bool>(),"finite test poison is invalid");
      for(bool master:{false,true}) {
        optimizer.values().zero_();gradient.values.zero_();gradient.connected.fill_(true);error.zero_();
        (master?optimizer.values():gradient.values).copy_(poisoned);
        const auto before=snapshot();portable_torch::synchronize(device);program.run();
        require(error.cpu().item<int>()==tide_device::optimizer_finite_error,"nonfinite optimizer lane was accepted");
        unchanged(before);++probes;
      }
      error.zero_();gradient.connected.zero_();optimizer.values().copy_(poisoned);gradient.values.copy_(poisoned);
      const auto before=snapshot();portable_torch::synchronize(device);program.run();
      require(error.cpu().item<int>()==0,"disconnected poison reached optimizer finite check");unchanged(before);++probes;
    }
    program.close();
  }
  std::cout<<"optimizer-finite: passed probes="<<probes<<" widths=9 tails=true nonfinite_atomic=true disconnected_poison=true\n";
}
} // namespace tide::device_online::test
