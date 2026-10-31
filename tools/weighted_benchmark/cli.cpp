#include "cli.h"
#include <cmath>
#include <iostream>
#include <sstream>
#include <limits>
namespace weighted_benchmark {
namespace {
int64_t integer(const std::string& text,const std::string& key,int64_t minimum=1) {
  size_t n=0;const auto value=std::stoll(text,&n);
  if(n!=text.size()||value<minimum)throw std::invalid_argument(key+" requires an int64 >= "+std::to_string(minimum));return value;
}
double number(const std::string& text,const std::string& key) {
  size_t n=0;const auto value=std::stod(text,&n);
  if(n!=text.size()||!std::isfinite(value)||value<0)throw std::invalid_argument(key+" requires a finite nonnegative number");return value;
}
bool boolean(const std::string& value,const std::string& key) {
  if(value!="true"&&value!="false")throw std::invalid_argument(key+" requires true or false");return value=="true";
}
std::vector<std::string> devices(const std::string& value) {
  std::vector<std::string> out;std::stringstream stream(value);std::string item;
  while(std::getline(stream,item,',')){if(item.empty())throw std::invalid_argument("empty device in comma-separated map");out.push_back(item);}
  if(out.empty()||value.back()==',')throw std::invalid_argument("empty device map");return out;
}
bool one(const std::string& value,std::initializer_list<const char*> choices) {
  for(const auto* c:choices)if(value==c)return true;return false;
}
}
Json Request::task() const {
  return {{"loss","physical-output-squared-v1"},{"loss_scale",loss_scale},{"optimizer","sgd"},
          {"lr",lr},{"momentum",momentum},{"weight_decay",weight_decay}};
}
void help() {
  std::cout<<"Standalone core-5 LibTorch experiment (no Python runtime scheduler).\n"
    "--fixture JSON --device cpu|npu:N|cuda:N --output-dir NEW\n"
    "--mode inference|training --schedule streaming|prefill --steps N --warmup N (default 1; 0 allowed)\n"
    "--dtype float16|float32|float64 --scoring-dtype float32|float64\n"
    "--implementation native|resident --placement native|mixed-a|mixed-b|mixed-c\n"
    "--threads N (--intraop-threads alias) --interop-threads N --workers N\n"
    "--packing auto|packed|scalar --return-packing scalar|packed\n"
    "--peer-device DEVICE (repeatable) or --peer-devices DEVICE,DEVICE\n"
    "--node-devices DEVICE,DEVICE,... (one owner per node, sharing constraints apply)\n"
    "--communication ordinary|blocking|cpu_bridge (process-scoped explicit copy policy)\n"
    "--validation strict|explore|none --return-mode complete|outputs|metrics\n"
    "  Minimal returns require validation=none; use a separate complete-return qualification.\n"
    "  Explore accepts only finite numeric mismatch with every discrete comparison passed.\n"
    "--resume CHECKPOINT_DIR --checkpoint-check true|false --checkpoint-max-bytes N\n"
    "  Each completed window atomically publishes output/checkpoint/latest.json; resume reads that boundary.\n"
    "--loss-scale X --lr X --momentum X --weight-decay X (explicit diagnostic training task)\n"
    "--check-roots [true|false] (separate small eager independent VJP diagnostic, <= 5M parameters)\n"
    "--resident-limits JSON --observe-work true|false (observation, not throughput qualification)\n";
}
Request parse_request(int argc,char** argv) {
  Request r;
  for(int i=1;i<argc;++i) {
    const std::string key=argv[i];if(key=="--help"){r.help=true;return r;}
    if(key=="--check-roots"){
      r.check_roots=true;
      if(i+1<argc&&(std::string(argv[i+1])=="true"||std::string(argv[i+1])=="false"))r.check_roots=boolean(argv[++i],key);
      continue;
    }
    if(i+1==argc)throw std::invalid_argument("missing value for "+key);const std::string value=argv[++i];
    if(key=="--fixture")r.input=value;else if(key=="--device")r.device=value;
    else if(key=="--output-dir")r.output=value;else if(key=="--resume")r.resume=value;
    else if(key=="--resident-limits")r.limits=value;else if(key=="--implementation")r.implementation=value;
    else if(key=="--mode")r.mode=value;else if(key=="--schedule")r.schedule=value;
    else if(key=="--dtype")r.precision=value;else if(key=="--scoring-dtype")r.scoring=value;
    else if(key=="--placement")r.placement=value;else if(key=="--packing")r.packing=value;
    else if(key=="--return-packing")r.return_packing=value;else if(key=="--validation")r.validation=value;
    else if(key=="--return-mode")r.return_mode=value;else if(key=="--communication")r.communication=value;
    else if(key=="--steps")r.steps=integer(value,key);else if(key=="--workers")r.workers=integer(value,key);
    else if(key=="--warmup")r.warmup=integer(value,key,0);
    else if(key=="--threads"||key=="--intraop-threads")r.threads=integer(value,key);
    else if(key=="--interop-threads")r.interop=integer(value,key);
    else if(key=="--checkpoint-max-bytes")r.checkpoint_bytes=integer(value,key);
    else if(key=="--checkpoint-check")r.checkpoint=boolean(value,key);
    else if(key=="--observe-work")r.observe_work=boolean(value,key);
    else if(key=="--peer-device")r.peers.push_back(value);
    else if(key=="--peer-devices")r.peers=devices(value);else if(key=="--node-devices")r.nodes=devices(value);
    else if(key=="--loss-scale")r.loss_scale=number(value,key);else if(key=="--lr")r.lr=number(value,key);
    else if(key=="--momentum")r.momentum=number(value,key);else if(key=="--weight-decay")r.weight_decay=number(value,key);
    else throw std::invalid_argument("unknown benchmark option "+key);
  }
  if(r.input.empty()||r.device.empty()||r.output.empty()||!one(r.implementation,{"native","resident"})||
     !one(r.mode,{"inference","training"})||!one(r.schedule,{"streaming","prefill"})||
     !one(r.precision,{"float16","float32","float64"}))throw std::invalid_argument("invalid bounded benchmark request; see --help");
  if(r.steps>std::numeric_limits<Index>::max()-r.warmup-2||r.threads>std::numeric_limits<int>::max()||
     r.interop>std::numeric_limits<int>::max())throw std::invalid_argument("window/thread request exceeds representable bounds");
  if(!one(r.validation,{"strict","explore","none"})||!one(r.return_mode,{"complete","outputs","metrics"}))
    throw std::invalid_argument("invalid validation or return-mode");
  if(!one(r.communication,{"ordinary","blocking","cpu_bridge"}))throw std::invalid_argument("invalid communication policy");
  if(r.return_mode!="complete"&&r.validation!="none")throw std::invalid_argument("minimal return requires validation=none; qualify complete records separately");
  if(r.return_mode!="complete"&&r.checkpoint)throw std::invalid_argument("exact complete-record resume check requires return-mode=complete");
  if(r.check_roots&&(r.validation=="none"||r.return_mode!="complete"||r.implementation!="native"))
    throw std::invalid_argument("--check-roots requires eager native, complete returns and strict/explore validation; resident has separate compiled root qualification");
  if(r.precision=="float64"&&((r.device!="cpu"&&r.device.rfind("cuda:",0)!=0)||r.implementation!="native"||!r.peers.empty()))
    throw std::invalid_argument("FP64 requires CPU or single CUDA eager; no implicit substitution");
  if(r.scoring.empty())r.scoring=r.precision=="float64"?"float64":"float32";
  if(!one(r.scoring,{"float32","float64"})||(r.implementation=="resident"&&r.scoring!="float32"))
    throw std::invalid_argument("invalid scoring dtype for requested implementation");
  if(r.precision=="float64"&&r.device.rfind("cuda:",0)==0&&r.scoring!="float64")throw std::invalid_argument("CUDA FP64 requires float64 scoring");
  if(!one(r.packing,{"auto","packed","scalar"})||!one(r.return_packing,{"scalar","packed"}))throw std::invalid_argument("invalid packing");
  if(r.implementation=="resident"&&(r.return_packing!="scalar"||r.placement!="native"||r.packing=="scalar"||r.workers!=1))
    throw std::invalid_argument("resident does not accept eager placement/scalar/host workers/return packing");
  if(r.implementation=="resident"&&(r.device=="cpu"||r.device=="auto"))throw std::invalid_argument("resident requires an indexed accelerator");
  if(r.observe_work&&r.implementation!="native")throw std::invalid_argument("eager packing observation does not apply to resident");
  r.input=std::filesystem::absolute(r.input);r.output=std::filesystem::absolute(r.output);
  if(!r.limits.empty())r.limits=std::filesystem::absolute(r.limits);
  if(!r.resume.empty())r.resume=std::filesystem::absolute(r.resume);
  return r;
}
}  // namespace weighted_benchmark
