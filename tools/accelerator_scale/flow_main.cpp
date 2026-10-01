#include "flow_config.h"
#include <ATen/Parallel.h>
#include <cstdlib>
#include <filesystem>
#include <iostream>

int main(int argc,char** argv) {
  using namespace accelerator_scale;bool npu=false;int result=0;
  try {
    const auto begin=pdg_scale::Clock::now();auto config=flows::parse(argc,argv);
    if(config.model.runtime.help) {
      portable_torch::print_usage(std::cout,argv[0]);
      std::cout<<"Complete independent window: --topology FILE --family pdg|timed-dag|settle\n"
        "--flow cpu|mixed|device --scheduler streaming|frontier|settle|resident|bounded-eager|bounded-replay\n"
        "--training 0|1 --optimizer sgd|adamw --devices 1..16 --partition locality|memory\n"
        "--dtype float64|float32|float16 --read-dtype float64|float32\n"
        "--read-device cpu|model --control-device cpu|model --ranking-device cpu|model --event-device cpu|model\n"
        "--iterations N --iteration-warmup N --workspace-gib N --warmup 0\n"
        "--workers N --head-workers N --threads N --backward-threads N --optimizer-threads N\n"
        "--profile-step N --profile-output NEW: separate instrumented whole-window CANN pass.\n"
        "--check 1 runs independent full-observable/VJP/update gates only (small shapes).\n"
        "Each measured iteration starts with empty graph state; training weights/optimizer persist.\n";
      return 0;
    }
    if(config.model.runtime.device_spec.rfind("npu",0)==0 && !std::getenv("TASK_QUEUE_ENABLE"))
      if(setenv("TASK_QUEUE_ENABLE","0",0))throw std::runtime_error("cannot set NPU task queue policy");
    const auto device=portable_torch::resolve_device(config.model.runtime);
    npu=device.type()==c10::DeviceType::PrivateUse1;
    const auto topology=flows::read_topology(config.model.topology);
    flows::validate(config,device,topology);
    at::set_num_threads(config.model.threads);at::set_num_interop_threads(1);
    config.runtime_setup_seconds=pdg_scale::seconds(begin);
    if(!std::filesystem::create_directories(config.model.runtime.output_dir))
      throw std::invalid_argument("new output directory required");
    if(config.model.check) {
      const auto contexts=initialize_devices(device,config.devices);
      flows::check(config,topology,device);
    } else flows::benchmark(config,topology,device);
  } catch(const std::exception& error){std::cerr<<error.what()<<'\n';result=1;}
  if(npu)try{finalize();}catch(const std::exception& error){std::cerr<<error.what()<<'\n';result=1;}
  return result;
}
