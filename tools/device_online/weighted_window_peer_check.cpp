// Core-5 finite-cut chain rule with one graph physically placed on two devices.
// Independent CPU schedules/autograd supply roots, never candidate work lists.
#include "weighted_window_check.h"
#include "weighted_flow_fixture.h"
#include "../../cpp/test/weighted_graph_fixture.h"
#include "portable_torch/runtime.hpp"
#include <ATen/Parallel.h>
#include <iostream>
namespace {
using namespace tide;
using namespace tide::device_online;
std::vector<at::Device> owners(const Graph& graph,at::Device device,bool all_remote=false){
  std::vector<at::Device> result;
  for(size_t v=0;v<graph.nodes.size();++v)result.emplace_back(device.type(),all_remote?1:(v%2));
  return result;
}
void run(at::Device device){
  int count=0;
  for(const std::string topology:{"hb","nonaligned","feedback"}){
    auto g=weighted_check::graph(topology,"heterogeneous");
    for(size_t r=0;r<g.regions.size();++r)g.regions[r].read_mode=std::vector<std::string>{"content","old","proposal"}[r%3];
    g.compile();auto m=weighted_check::model(g,"heterogeneous",at::kCPU,at::kFloat);
    auto xs=weighted_check::inputs(g,at::kCPU,at::kFloat);
    xs.erase(xs.begin()+4);for(auto& x:xs){if(x.batch==1&&x.port==0)x.position=0;
      if(x.batch==1&&x.time==1)x.value=at::zeros_like(x.value);}
    // Keep both trainable shared uses and frozen owners in the same graph.
    if(topology=="feedback")for(auto& r:m.regions)for(auto& [name,x]:r.extra)x.set_requires_grad(false);
    auto placement=owners(g,device,topology=="nonaligned");
    window_check::run_window(device,g,m,xs,"placed/"+topology,true,topology!="feedback",topology=="hb",placement);++count;
  }
  auto [g,m]=weighted_flow_check::configured(weighted_flow_check::configurations()[0],true,true,true);
  auto xs=weighted_check::inputs(g,at::kCPU,at::kFloat);
  for(auto& x:xs)x.value=x.value*at::tensor({.3f,1.f,1.f});xs[0].value=at::zeros_like(xs[0].value);
  window_check::run_window(device,g,m,xs,"placed/global-attention",true,true,true,owners(g,device));++count;
  for(const std::string topology:{"hb","feedback"}){
    auto graph=weighted_check::graph(topology,"heterogeneous");
    window_check::run_retained(device,topology,"proposal",owners(graph,device));
  }
  std::cout<<"weighted-node-window: passed configurations="<<count
    <<" root_modes=12 bridges=2 owner_local_vjp=true closed_forward=true empty_windows=true\n";
}
}
int main(int argc,char** argv){portable_torch::RuntimeSession runtime;
  try{auto args=portable_torch::parse_cli(argc,argv,true);
    if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("weighted node window requires explicit FP32 accelerator");
    auto device=portable_torch::resolve_device(args);if(device.index()!=0)throw std::invalid_argument("checker requires coordinator 0 and peer 1");
    at::set_num_threads(1);at::set_num_interop_threads(1);run(device);return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
