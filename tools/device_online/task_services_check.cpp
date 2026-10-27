// Device-commanded numerical phases, skipped windows and repeated submissions.
#include "task_services.h"
#include "portable_torch/runtime.hpp"
#include <ATen/Parallel.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <iostream>
using namespace tide::device_online;
int main(int argc,char** argv){portable_torch::RuntimeSession runtime;
  try{
    auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("services check requires explicit accelerator FP32");
    const auto device=portable_torch::resolve_device(args);at::set_num_threads(1);at::set_num_interop_threads(1);at::NoGradGuard guard;
    auto opts=at::TensorOptions().device(device).dtype(at::kLong);
    auto count=at::zeros({1},opts),limit=at::zeros_like(count),one=at::ones_like(count),branch=at::zeros({1},opts.dtype(at::kInt));
    auto condition=at::zeros({1},opts.dtype(at::kBool));auto host=at::tensor({.2f,-.3f,.7f,.1f},at::kFloat).reshape({2,2});
    auto x=host.to(device),normal=at::empty_like(x),scaled=at::empty_like(x),y=at::empty_like(x),factor=at::full_like(x,.7);
    DeviceProgram coordinator(device);coordinator.limit_workspace(64*1024*1024);TaskServices services(coordinator,device,64*1024*1024);
    const auto head=coordinator.label(),body=coordinator.label(),end=coordinator.label();coordinator.mark(head);
    coordinator.less(count,limit,condition);coordinator.cast_index(condition,branch);coordinator.branch(branch,{end,body});coordinator.mark(body);
    auto& first=services.begin();first.rms_norm(x,.3,normal);
    auto& second=services.begin();second.multiply(normal,factor,scaled);
    auto& third=services.begin();third.rms_norm(scaled,.2,y);services.end(third);
    services.end(second);services.end(first);
    coordinator.copy(x,y);coordinator.add(count,one);coordinator.branch(branch,{head});coordinator.mark(end);services.append_stop();
    const auto workspace=services.workspace_bytes();
#if !TIDE_RESIDENT_CUDA
    int64_t summed=0;for(const auto* p:services.programs())summed+=p->workspace_bytes();
    if(!(workspace>0&&summed>workspace))throw std::runtime_error("services did not exercise shared workspace");
#endif
    services.finish();coordinator.finish();
    for(int loops:{3,0,5}){
      x.copy_(host);count.zero_();limit.fill_(loops);c10::impl::VirtualGuardImpl(device.type()).synchronizeDevice(device.index());
      services.submit();coordinator.run(10000);services.wait(10000);
      auto expected=host.clone();for(int i=0;i<loops;++i){expected=expected/at::sqrt(expected.square().mean(1,true)+.3)*.7;
        expected=expected/at::sqrt(expected.square().mean(1,true)+.2);}
      if(count.item<int64_t>()!=loops||!at::allclose(x.cpu(),expected,2e-5,2e-6))throw std::runtime_error("service cycle/window differs from independent recurrence");
    }
    coordinator.close();services.close();std::cout<<"task-services: passed repeated_windows=true empty_window=true nested_levels=3 shared_workspace_bytes="<<workspace<<'\n';return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
