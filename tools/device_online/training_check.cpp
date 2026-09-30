#include "training_test.h"
#include "portable_torch/runtime.hpp"
#include "../../cpp/bench/streaming.h"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <iostream>

namespace {
using namespace tide;using namespace tide::device_online;

}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("public training check requires explicit NPU FP32");
    auto device=portable_torch::resolve_device(args);at::set_num_threads(1);at::set_num_interop_threads(1);int cases=0;
    for(int shape:{0,3})for(bool prefill:{false,true})for(auto kind:{ResidentOptimizerKind::sgd,ResidentOptimizerKind::adamw})for(auto dtype:{at::kFloat,at::kDouble}) {
      try{test::train_trajectory(device,test::retained_fixture(shape,0,3),prefill,kind,dtype);++cases;}
      catch(...){std::cerr<<"public training shape="<<shape<<" prefill="<<prefill<<" kind="<<int(kind)<<" dtype="<<dtype<<'\n';throw;}
    }
    test::train_trajectory(device,test::retained_fixture(0,0,257),true,ResidentOptimizerKind::adamw,at::kFloat);++cases;
    test::train_trajectory(device,test::retained_fixture(3,1,3),true,ResidentOptimizerKind::sgd,at::kDouble);++cases;
    test::train_failures(device);
    std::cout<<"public-resident-training: passed trajectories="<<cases<<" windows="<<cases*16<<" updates="<<cases*4
      <<" CPU=FP32_FP64 checkpoint_resume=true lifecycle=true scope=single_NPU_HARD_subset\n";
    runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
