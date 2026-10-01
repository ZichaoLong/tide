#include "cann_sequence.h"
#include "portable_torch/runtime.hpp"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <iostream>
namespace {
using namespace tide::device_online;
void require(bool ok){if(!ok)throw std::runtime_error("device program sequence check failed");}
template<class F>void reject(F f){bool refused=false;try{f();}catch(const std::logic_error&){refused=true;}require(refused);}
void check(at::Device device) {
  at::NoGradGuard guard;
  const auto options=at::TensorOptions().device(device).dtype(at::kLong);
  auto value=at::zeros({1},options),one=at::ones_like(value),iteration=at::zeros_like(value),limit=at::zeros_like(value);
  CannSequence sequence(device,8,4*1024*1024);reject([&]{sequence.finish();});reject([&]{sequence.submit();});
  constexpr int64_t block=512,initial=(int64_t(1)<<55)+17;
  for(int i=0;i<8;++i) {
    auto& p=sequence.append();auto predicate=at::zeros({1},options.dtype(at::kBool)),index=at::zeros({1},options.dtype(at::kInt));
    const auto body=p.label(),done=p.label();p.less(iteration,limit,predicate);p.cast_index(predicate,index);p.branch(index,{done,body});
    p.mark(body);for(int j=0;j<block;++j)p.add(value,one);p.add(iteration,one);p.mark(done);
  }
  reject([&]{sequence.append();});require(sequence.size()==8);sequence.finish();reject([&]{sequence.append();});
  for(int loops:{0,3,8,2}) {
    value.fill_(initial);iteration.zero_();limit.fill_(loops);portable_torch::synchronize(device);sequence.run();
    require(value.cpu().item<int64_t>()==initial+block*loops);require(iteration.cpu().item<int64_t>()==loops);
  }
  sequence.close();sequence.close();reject([&]{sequence.submit();});
  // A one-program chain needs no notification and remains replayable.
  CannSequence single(device,1,4*1024*1024);single.append().add(value,one);single.finish();
  portable_torch::synchronize(device);single.run();require(value.cpu().item<int64_t>()==initial+2*block+1);single.close();
  std::cout<<"device-sequence: passed programs=8 static_add_tasks=4096 replays=4 exact_int64=true device_completion=true capacity_refused=true single_program=true\n";
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("sequence check requires explicit NPU FP32 runtime");
    const auto d=portable_torch::resolve_device(args);if(d.type()!=c10::DeviceType::PrivateUse1)throw std::invalid_argument("NPU required");
    at::set_num_threads(1);at::set_num_interop_threads(1);check(d);runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
