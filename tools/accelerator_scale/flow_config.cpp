#include "flow_config.h"
#include <set>
#include <stdexcept>
#include <filesystem>

namespace accelerator_scale::flows {
Config parse(int argc,char** argv) {
  Config c;std::vector<char*> common{argv[0]};std::set<std::string> seen;
  std::map<std::string,std::string*> strings{{"--family",&c.family},{"--flow",&c.flow},
    {"--scheduler",&c.scheduler},{"--optimizer",&c.optimizer},{"--partition",&c.partition},
    {"--read-device",&c.scoring.read_device},{"--control-device",&c.scoring.control_device},
    {"--ranking-device",&c.ranking},{"--event-device",&c.events},{"--profile-output",&c.profile.output}};
  std::map<std::string,Index*> ints{{"--devices",&c.devices},{"--iterations",&c.iterations},
    {"--iteration-warmup",&c.iteration_warmup},{"--workspace-gib",&c.workspace_gib},
    {"--backward-threads",&c.backward_threads},{"--optimizer-threads",&c.optimizer_threads},{"--profile-step",&c.profile.step}};
  for(int i=1;i<argc;++i) {
    const std::string key=argv[i];
    if(!strings.count(key) && !ints.count(key) && key!="--training" && key!="--read-dtype") {
      common.push_back(argv[i]);continue;
    }
    if(!seen.insert(key).second || ++i==argc)throw std::invalid_argument("duplicate/missing "+key);
    const std::string value=argv[i];
    if(strings.count(key)){*strings[key]=value;continue;}
    if(key=="--read-dtype") {
      if(value!="float32" && value!="float64")throw std::invalid_argument("Read requires FP32 or FP64");
      c.scoring.dtype=value=="float64"?at::kDouble:at::kFloat;continue;
    }
    if(value.empty() || value.find_first_not_of("0123456789")!=std::string::npos)
      throw std::invalid_argument("nonnegative integer required: "+key);
    const auto n=std::stoll(value);
    if(ints.count(key))*ints[key]=n;
    else {if(n>1)throw std::invalid_argument("training requires0|1");c.training=n;}
  }
  c.model=pdg_scale::parse(common.size(),common.data());c.model.grad=c.training;
  if(c.model.runtime.help)return c;
  if(c.flow=="device") {
    if(!seen.count("--read-device"))c.scoring.read_device="model";
    if(!seen.count("--control-device"))c.scoring.control_device="model";
    if(!seen.count("--ranking-device"))c.ranking="model";
    if(!seen.count("--event-device"))c.events="model";
  }
  if(c.scheduler.empty())c.scheduler=c.flow=="device"?"bounded-replay":c.flow=="mixed"?"resident":
    c.family=="settle"?"settle":c.family=="timed-dag"?"frontier":"streaming";
  if(c.model.runtime.dtype==at::kDouble && !seen.count("--read-dtype"))c.scoring.dtype=at::kDouble;
  c.model.runtime.allow_npu_float16=c.model.runtime.dtype==at::kHalf;
  return c;
}
void validate(const Config& c,at::Device device,const Topology& topology) {
  if(c.family!="pdg" && c.family!="timed-dag" && c.family!="settle")throw std::invalid_argument("unknown graph family");
  if(c.family=="settle" && !topology.rank_aligned)throw std::invalid_argument("Settle requires rank-aligned delays");
  if(c.flow!="cpu" && c.flow!="mixed" && c.flow!="device")throw std::invalid_argument("unknown complete flow");
  if((c.flow=="cpu")!=device.is_cpu())throw std::invalid_argument("flow/device request mismatch");
  if(c.devices<1 || c.devices>16 || (device.is_cpu() && c.devices!=1)
      || c.iterations<2 || c.iterations>100 || c.iteration_warmup>=c.iterations
      || c.workspace_gib<1 || c.workspace_gib>32768 || c.model.steps>32 || c.model.batch>512
      || c.backward_threads<1 || c.backward_threads>160 || c.optimizer_threads<1 || c.optimizer_threads>160
      || (c.optimizer!="sgd" && c.optimizer!="adamw"))throw std::invalid_argument("invalid complete-flow capacity");
  const std::set<std::string> schedules{"streaming","frontier","settle","resident","bounded-eager","bounded-replay"};
  if(!schedules.count(c.scheduler) || (c.scheduler=="settle" && c.family!="settle"))throw std::invalid_argument("invalid family scheduler");
  if(c.flow=="mixed" && c.scheduler!="resident")throw std::invalid_argument("mixed flow requires resident host dispatch");
  if(c.flow=="device" && c.scheduler!="bounded-replay" && c.scheduler!="bounded-eager")
    throw std::invalid_argument("device flow requires bounded eager/replay schedule");
  if(c.scheduler=="bounded-replay" && device.type()!=c10::DeviceType::PrivateUse1)
    throw std::invalid_argument("bounded replay requires NPU");
  if(c.scheduler.rfind("bounded-",0)==0 && (c.scoring.dtype!=at::kFloat || c.model.runtime.dtype==at::kDouble
      || c.scoring.read_device!="model" || c.scoring.control_device!="model" || c.ranking!="model" || c.events!="model"))
    throw std::invalid_argument("bounded schedule requires model FP32 controls and FP32/FP16 payload");
  c.scoring.validate(device,true);
  for(const auto& value:{c.ranking,c.events})if(value!="cpu" && value!="model")throw std::invalid_argument("invalid dispatch placement");
  if(c.partition!="locality" && c.partition!="memory")throw std::invalid_argument("invalid partition policy");
  if(c.model.runtime.device_spec=="auto" || c.model.emission!="row" || c.model.warmup!=0
      || c.model.profile || c.model.operator_profile || c.model.work_count)
    throw std::invalid_argument("complete flow requires explicit backend, row emission, token warmup0 and separate profiling");
  if(!device.is_cpu() && (c.model.head_workers!=1 || c.backward_threads!=1 || c.optimizer_threads!=1))
    throw std::invalid_argument("accelerator flow requires head/backward/optimizer threads1");
  if(c.model.runtime.dtype!=at::kDouble && c.model.runtime.dtype!=at::kFloat && c.model.runtime.dtype!=at::kHalf)
    throw std::invalid_argument("complete-flow payload requires FP64/FP32/FP16");
  if(c.profile.enabled()) {
    if(device.type()!=c10::DeviceType::PrivateUse1 || c.profile.output.empty() || c.profile.step>=c.iterations
        || c.model.check || std::filesystem::exists(c.profile.output))
      throw std::invalid_argument("window profiling requires NPU benchmark, valid iteration and new output path");
  } else if(!c.profile.output.empty())throw std::invalid_argument("profile output requires profile step");
  if(c.model.projection_layout!="input")throw std::invalid_argument("complete-flow fixture requires input projection layout");
  if(c.scheduler.rfind("bounded-",0)==0 && (c.model.full_autograd!="replay" || c.model.aggregate_autograd!="replay"
      || c.model.packed_sources || c.model.batch_next || c.model.parallel_regions || c.model.compact_events
      || c.model.defer_state_release || c.model.attention_packing!="exact" || c.model.fiber_cache!="cloned"
      || c.model.attention_layout!="event" || c.model.fiber_pooling!="event"))
    throw std::invalid_argument("bounded tensor program cannot accept eager scheduler/kernel optimization flags");
  if(c.model.check && (c.model.width>64 || c.model.batch>8 || c.model.steps>6))throw std::invalid_argument("complete flow check requires small tensors");
}
Options options(const Config& c,bool trace) {
  Options o;o.trace=trace;o.workers=c.model.workers;o.packed=c.model.packed;
  o.parallel_regions=c.model.parallel_regions;o.compact_events=c.model.compact_events;
  o.defer_state_release=c.model.defer_state_release;o.packed_sources=c.model.packed_sources;
  o.batch_next=c.model.batch_next;o.full_autograd=c.model.full_autograd;o.aggregate_autograd=c.model.aggregate_autograd;
  return o;
}
}  // namespace accelerator_scale::flows
