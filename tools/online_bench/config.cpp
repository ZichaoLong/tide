#include "consumer.h"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace tide_flow {
Config parse(int argc,char** argv) {
  Config c;std::vector<char*> forwarded{argv[0]};bool preset=false;
  const std::vector<std::string> limits={"queue","arrivals","outputs","trace","stages","workspace-bytes",
    "full-chunk-rows","emission-chunk-rows","aggregate-chunk-rows","attention-chunk-rows","attention-key-rows",
    "kv-rows","kv-trace-rows","max-repeat-ticks","retained-bytes","backward-bytes","optimizer-bytes",
    "program-workspace-bytes","reverse-chunk-rows"};
  auto integer=[](const std::string& value) {
    size_t used=0;const auto result=std::stoll(value,&used);
    if(used!=value.size()||result<0)throw std::invalid_argument("invalid nonnegative integer option");
    return result;
  };
  for(int i=1;i<argc;++i) {
    std::string arg=argv[i];auto equal=arg.find('=');
    const auto key=arg.substr(0,equal);auto value=equal==std::string::npos?std::string():arg.substr(equal+1);
    if(arg=="--training"){c.training=true;continue;}
    if(arg=="--diagnostics"){c.diagnostics=true;continue;}
    const auto capacity=key.rfind("--resident-",0)==0?key.substr(11):std::string();
    const bool limit=std::find(limits.begin(),limits.end(),capacity)!=limits.end();
    const bool known=key=="--packet"||key=="--family"||key=="--schedule"||key=="--preset"||key=="--optimizer"
      ||key=="--steps"||key=="--warmup"||key=="--windows-per-step"||key=="--threads"||key=="--parameter-budget"
      ||key=="--read"||key=="--control"||key=="--selection"||key=="--events"||key=="--scoring-dtype"
      ||key=="--devices"||key=="--owner-policy"||key=="--chunk-policy"||key=="--head-workspace-bytes"||key=="--device-memory-bytes"||limit;
    if(!known){forwarded.push_back(argv[i]);continue;}
    if(equal==std::string::npos){if(++i==argc)throw std::invalid_argument("missing option value");value=argv[i];}
    if(key=="--packet")c.packet=value;else if(key=="--family")c.family=value;
    else if(key=="--schedule")c.schedule=value;else if(key=="--preset"){c.placement.preset=value;preset=true;}
    else if(key=="--optimizer")c.optimizer=value;
    else if(key=="--read")c.placement.read=value;else if(key=="--control")c.placement.control=value;
    else if(key=="--selection")c.placement.selection=value;else if(key=="--events")c.placement.events=value;
    else if(key=="--scoring-dtype")c.placement.scoring_dtype=value;
    else if(key=="--steps")c.steps=integer(value);else if(key=="--warmup")c.warmup=integer(value);
    else if(key=="--windows-per-step")c.windows=integer(value);else if(key=="--threads")c.threads=integer(value);
    else if(key=="--parameter-budget")c.parameter_budget=integer(value);
    else if(key=="--head-workspace-bytes")c.head_workspace_bytes=integer(value);
    else if(key=="--device-memory-bytes")c.device_memory_bytes=integer(value);
    else if(key=="--devices")c.devices=integer(value);
    else if(key=="--owner-policy")c.owner_policy=value;
    else if(key=="--chunk-policy")c.chunk_policy=value;
    else if(limit)c.resident_limits.emplace(capacity,integer(value));
  }
  c.runtime=portable_torch::parse_cli(forwarded.size(),forwarded.data(),true);
  if(c.runtime.help)return c;
  if(c.packet.empty()||c.runtime.output_dir.empty()||!preset
      ||(c.family!="pdg"&&c.family!="timed-dag"&&c.family!="settle")
      ||(c.schedule!="streaming"&&c.schedule!="prefill")
      ||(c.optimizer!="sgd"&&c.optimizer!="adamw")||c.steps<1||c.windows<1||c.threads<1||c.threads>1024
      ||c.steps>1000000||c.warmup>1000000||c.windows>1000000||c.parameter_budget<1||c.head_workspace_bytes<1)
    throw std::invalid_argument("explicit packet/output-dir/family/preset/schedule and positive bounded run limits required");
  if(c.devices<1||c.devices>16||(c.owner_policy!="memory"&&c.owner_policy!="locality")
      ||(c.chunk_policy!="conservative"&&c.chunk_policy!="aggressive"))throw std::invalid_argument("invalid device/owner/chunk policy");
  if(c.placement.preset!="resident"&&(c.devices!=1||!c.resident_limits.empty()||c.owner_policy!="locality"||c.chunk_policy!="conservative"||c.head_workspace_bytes!=4LL*1024*1024*1024||c.device_memory_bytes))
    throw std::invalid_argument("resident capacities and placement require resident preset");
  if(c.placement.preset=="resident"&&c.runtime.dtype!=at::kFloat&&c.runtime.dtype!=at::kHalf)
    throw std::invalid_argument("resident consumer requires FP32/FP16 payload");
  if((c.runtime.dtype!=at::kFloat&&c.runtime.dtype!=at::kDouble&&c.runtime.dtype!=at::kHalf)
      ||(c.training&&c.runtime.dtype==at::kHalf&&c.placement.preset!="resident"))
    throw std::invalid_argument("eager consumer FP16 training requires a qualified master optimizer");
  c.runtime.allow_npu_float16=!c.training||c.placement.preset=="resident";
  return c;
}
} // namespace tide_flow
