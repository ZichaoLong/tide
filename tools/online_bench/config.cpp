#include "consumer.h"
#include <algorithm>
#include <cmath>
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
    if(arg=="--phase-timing"){c.phase_timing=true;continue;}
    if(arg=="--packed-sources"){c.packed_sources=true;continue;}
    if(arg=="--batch-next"){c.batch_next=true;continue;}
    if(arg=="--auto-sample-chunks"){c.auto_sample_chunks=true;continue;}
    const auto capacity=key.rfind("--resident-",0)==0?key.substr(11):std::string();
    const bool limit=std::find(limits.begin(),limits.end(),capacity)!=limits.end();
    const bool known=key=="--packet"||key=="--family"||key=="--schedule"||key=="--preset"||key=="--optimizer"
      ||key=="--steps"||key=="--warmup"||key=="--windows-per-step"||key=="--threads"||key=="--workers"||key=="--parameter-budget"||key=="--sample-chunk-rows"
      ||key=="--read"||key=="--control"||key=="--selection"||key=="--events"||key=="--scoring-dtype"||key=="--loss-scale"
      ||key=="--devices"||key=="--owner-policy"||key=="--owner-map"||key=="--chunk-policy"||key=="--head-workspace-bytes"||key=="--device-memory-bytes"||key=="--resident-context-bytes"||limit;
    if(!known){forwarded.push_back(argv[i]);continue;}
    if(equal==std::string::npos){if(++i==argc)throw std::invalid_argument("missing option value");value=argv[i];}
    if(key=="--packet")c.packet=value;else if(key=="--family")c.family=value;
    else if(key=="--schedule")c.schedule=value;else if(key=="--preset"){c.placement.preset=value;preset=true;}
    else if(key=="--optimizer")c.optimizer=value;
    else if(key=="--loss-scale") {
      size_t used=0;c.loss_scale=std::stod(value,&used);
      if(used!=value.size()||!std::isfinite(c.loss_scale)||c.loss_scale<=0)
        throw std::invalid_argument("loss-scale must be positive and finite");
    }
    else if(key=="--read")c.placement.read=value;else if(key=="--control")c.placement.control=value;
    else if(key=="--selection")c.placement.selection=value;else if(key=="--events")c.placement.events=value;
    else if(key=="--scoring-dtype")c.placement.scoring_dtype=value;
    else if(key=="--steps")c.steps=integer(value);else if(key=="--warmup")c.warmup=integer(value);
    else if(key=="--windows-per-step")c.windows=integer(value);else if(key=="--threads")c.threads=integer(value);
    else if(key=="--workers")c.workers=integer(value);
    else if(key=="--sample-chunk-rows")c.sample_chunk_rows=integer(value);
    else if(key=="--parameter-budget")c.parameter_budget=integer(value);
    else if(key=="--head-workspace-bytes")c.head_workspace_bytes=integer(value);
    else if(key=="--resident-context-bytes")c.context_memory_bytes=integer(value);
    else if(key=="--device-memory-bytes")c.device_memory_bytes=integer(value);
    else if(key=="--devices")c.devices=integer(value);
    else if(key=="--owner-policy")c.owner_policy=value;
    else if(key=="--owner-map") {
      if(value.empty()||value.back()==',')throw std::invalid_argument("invalid consumer owner map");
      c.owner_map.clear();std::istringstream in(value);std::string part;
      while(std::getline(in,part,',')) {
        if(part.empty()||part.size()>2||part.find_first_not_of("0123456789")!=std::string::npos)
          throw std::invalid_argument("invalid consumer owner map");
        auto d=integer(part);if(d>15)throw std::invalid_argument("invalid consumer owner map");c.owner_map.push_back(d);
      }
    }
    else if(key=="--chunk-policy")c.chunk_policy=value;
    else if(limit)c.resident_limits.emplace(capacity,integer(value));
  }
  c.runtime=portable_torch::parse_cli(forwarded.size(),forwarded.data(),true);
  if(c.runtime.help)return c;
  if(c.packet.empty()||c.runtime.output_dir.empty()||!preset
      ||(c.family!="pdg"&&c.family!="timed-dag"&&c.family!="settle")
      ||(c.schedule!="streaming"&&c.schedule!="prefill")
      ||(c.optimizer!="sgd"&&c.optimizer!="adamw")||c.steps<1||c.windows<1||c.threads<1||c.threads>1024||c.workers<1||c.workers>1024
      ||c.steps>1000000||c.warmup>1000000||c.windows>1000000||c.parameter_budget<1||c.head_workspace_bytes<1)
    throw std::invalid_argument("explicit packet/output-dir/family/preset/schedule and positive bounded run limits required");
  if(c.devices<1||c.devices>16||(c.owner_policy!="memory"&&c.owner_policy!="locality")
      ||(c.chunk_policy!="conservative"&&c.chunk_policy!="aggressive"))throw std::invalid_argument("invalid device/owner/chunk policy");
  if(c.placement.preset!="resident"&&(!c.resident_limits.empty()||c.context_memory_bytes))
    throw std::invalid_argument("resident capacities require resident preset");
  if(c.placement.preset=="resident"&&c.runtime.dtype!=at::kFloat&&c.runtime.dtype!=at::kHalf)
    throw std::invalid_argument("resident consumer requires FP32/FP16 payload");
  if(c.placement.preset=="resident"&&(c.workers!=1||c.packed_sources||c.batch_next))
    throw std::invalid_argument("host workers/packed-sources/batch-next require an eager native consumer");
  if(c.runtime.dtype!=at::kFloat&&c.runtime.dtype!=at::kDouble&&c.runtime.dtype!=at::kHalf)
    throw std::invalid_argument("consumer requires FP16/FP32/FP64 payload");
  if(c.loss_scale!=1&&(!c.training||c.runtime.dtype!=at::kHalf||c.placement.preset=="resident"))
    throw std::invalid_argument("nonunit loss-scale requires eager FP16 training");
  c.runtime.allow_npu_float16=true;
  return c;
}
} // namespace tide_flow
