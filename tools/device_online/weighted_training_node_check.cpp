// Public placed-node training: independent CPU schedules, roots and optimizers.
#include "weighted_training_check.h"
#include "optimizer_layout.h"
#include "weighted_precision_reference.h"
#include "../../cpp/test/weighted_graph_fixture.h"
#include "../../cpp/bench/streaming.h"
#include "tide/greedy.h"
#include "portable_torch/runtime.hpp"
#include <ATen/Parallel.h>
#include <iostream>
namespace {
using namespace tide;using namespace tide::device_online;using namespace weighted_training_check;
std::vector<at::Device> placement(const Graph& graph,at::Device device){
  std::vector<at::Device> out;for(size_t v=0;v<graph.nodes.size();++v)out.emplace_back(device.type(),v%2);return out;
}
void refusal(at::Device device,at::ScalarType dtype){
  auto g=weighted_check::graph("fc","kv");auto model=weighted_check::model(g,"kv",at::kCPU,dtype);
  auto reference=dtype==at::kHalf?weighted_precision_check::reference_model(model):model;
  Continuation q;q.identity=g.identity;q.batch_size=2;OptimizerGroup group;
  for(const auto& owner:model.parameters().owners())group.parameters.push_back(owner.canonical);
  group.lr=dtype==at::kHalf?1e10:1e38;group.momentum=.3;at::NoGradGuard ng;
  WeightedTrainingSession owner(g,model,q,device,ResidentOptimizerKind::sgd,{group},limits(),placement(g,device));
  auto before=owner.checkpoint();auto xs=weighted_check::inputs(g,at::kCPU,dtype);
  auto w=owner.advance(xs,10);auto cut=owner.result().continuation;auto root=seed(w,"outputs");
  for(auto& [name,x]:root.fields)x.values=x.values*1e6;
  const auto grad=owner.backward({root});
  // Independently establish that this finite gradient cannot publish to the
  // requested storage dtype; no candidate status is used as the premise.
  const auto proposed=before.state.values.to(at::kDouble)-group.lr*grad.parameters.values.cpu().to(at::kDouble);
  require(!at::isfinite(proposed.to(dtype)).all().item<bool>(),"publication refusal premise did not overflow storage");
  auto step=owner.step();
  require(!step.applied&&step.refusal_code==tide_device::optimizer_finite_error,
      "placed optimizer did not jointly refuse overflow: applied="+std::to_string(step.applied)+
      " code="+std::to_string(step.refusal_code));
  owner.detach();auto after=owner.checkpoint();
  for(const auto& [name,x]:before.parameters)require(at::equal(x,after.parameters.at(name)),"rejected publication changed named parameter");
  for(const auto& pair:std::vector<std::pair<Tensor,Tensor>>{{before.state.values,after.state.values},{before.state.first,after.state.first},
      {before.state.second,after.state.second},{before.state.maximum,after.state.maximum},{before.state.steps,after.state.steps},{before.state.corrections,after.state.corrections}})
    require(at::equal(pair.first,pair.second),"rejected publication changed optimizer state");
  require(after.generation==before.generation,"rejected publication changed generation");
  tide_bench::compare(Result{cut},Result{after.continuation},true,at::kFloat,std::nullopt,0.,0.);
  auto next=weighted_check::inputs(g,at::kCPU,dtype,10);for(auto& x:next)x.position+=2;
  owner.advance(next,20);Options options;options.packed=false;
  auto expected=Greedy(g,reference,options).run(weighted_precision_check::reference_continuation(cut),weighted_precision_check::reference_inputs(next),20,20);
  tide_bench::compare(owner.result(),expected,true,dtype,std::nullopt,dtype==at::kHalf?2e-2:-1.,dtype==at::kHalf?1e-3:-1.);
  owner.detach();owner.close();
}
}
int main(int argc,char** argv){portable_torch::RuntimeSession runtime;
  try{auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||(args.dtype!=at::kFloat&&args.dtype!=at::kHalf))throw std::invalid_argument("placed training requires explicit FP32/FP16 accelerator");
    args.allow_npu_float16=true;auto d=portable_torch::resolve_device(args);
    if(d.index()!=0)throw std::invalid_argument("placed checker requires coordinator 0 and peer 1");
    at::set_num_threads(1);at::set_num_interop_threads(1);
    const std::vector<std::tuple<std::string,std::string,ResidentOptimizerKind,std::string>> cases{
      {"fc","kv",ResidentOptimizerKind::sgd,"none"},{"hb","heterogeneous",ResidentOptimizerKind::adamw,"none"},
      {"nonaligned","delta",ResidentOptimizerKind::sgd,"partial"},{"feedback","heterogeneous",ResidentOptimizerKind::adamw,"partial"},
      {"fc","affine",ResidentOptimizerKind::sgd,"all"}};
    for(const auto& [topology,family,kind,frozen]:cases)
      trajectory(d,topology,family,kind,frozen,args.dtype,placement(weighted_check::graph(topology,family),d));
    auto owners=placement(weighted_check::graph("fc","kv"),d);failures(d,args.dtype,owners);accumulation(d,args.dtype,owners);refusal(d,args.dtype);
    std::cout<<"weighted-node-training: passed configurations=5 updates=20 windows=3 relocated_resume=true common_refusal=true\n";return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
