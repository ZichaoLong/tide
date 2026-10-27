#include "weighted_window_check.h"
#include "../../cpp/test/weighted_graph_fixture.h"
#include "portable_torch/runtime.hpp"
#include <ATen/Parallel.h>
#include <iostream>
namespace tide::device_online::window_check {
void run(at::Device device,const std::string& topology,const std::string& read,bool split,bool retained){
  auto g=weighted_check::graph(topology,"heterogeneous");for(auto& r:g.regions)r.read_mode=read;g.compile();
  auto m=weighted_check::model(g,"heterogeneous",at::kCPU,at::kFloat);auto xs=weighted_check::inputs(g,at::kCPU,at::kFloat);
  if(topology=="hb"&&read=="content")for(const auto& owner:m.parameters().owners())owner.value.set_requires_grad(false);
  else if(topology=="feedback"&&read=="old")for(auto& r:m.regions)for(auto& [name,value]:r.extra)value.set_requires_grad(false);
  xs.erase(xs.begin()+4);for(auto& x:xs){if(x.batch==1&&x.port==0)x.position=0;
    if(x.batch==1&&x.time==1)x.value=at::zeros_like(x.value);}
  run_window(device,std::move(g),std::move(m),std::move(xs),topology+"/"+read,split,read!="old",retained);
}
}

int main(int argc,char** argv){portable_torch::RuntimeSession runtime;
  try{bool retained=false;std::string selected;std::vector<char*> forwarded{argv[0]};
    for(int i=1;i<argc;++i){const std::string arg=argv[i];if(arg=="--saved-window"){retained=true;continue;}if(arg.rfind("--case=",0)==0){
      if(!selected.empty())throw std::invalid_argument("duplicate --case");selected=arg.substr(7);
      if(selected.empty())throw std::invalid_argument("empty --case");
    }else forwarded.push_back(argv[i]);}
    auto args=portable_torch::parse_cli(forwarded.size(),forwarded.data(),true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);
      std::cout<<"  --case=TOPOLOGY/READ  Run one unchanged fixture for diagnosis\n  --saved-window  Preserve actual operands after closing forward\n";return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("weighted window requires explicit accelerator FP32");
    auto device=portable_torch::resolve_device(args);at::set_num_threads(1);at::set_num_interop_threads(1);
    int cases=0;
    for(const std::string topology:{"fc","hb","nonaligned","feedback"})for(const std::string read:{"content","old","proposal"}){
      if(!selected.empty()&&selected!=topology+"/"+read)continue;++cases;
      try{tide::device_online::window_check::run(device,topology,read,read!="content",retained);}catch(...){std::cerr<<topology<<' '<<read<<'\n';throw;}}
    if(!cases)throw std::invalid_argument("unknown --case="+selected);
    std::cout<<(retained?"weighted-saved-window: passed configurations=":"weighted-window: passed configurations=")<<cases<<" complete_window=true device_reverse_loop=true independent_roots=true\n";return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
