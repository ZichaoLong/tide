#include "training_test.h"
#include "portable_torch/runtime.hpp"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <array>
#include <iostream>

namespace {
using namespace tide;using namespace tide::device_online;
const std::array<std::string,5> kinds{"sum","mean","weighted_mean","active_softmax","all_softmax"};
void coefficients(test::Fixture& f,int kind) {
  for(size_t n=0;n<f.graph.nodes.size();++n) {
    auto& node=f.graph.nodes[n];if(node.identity)continue;
    const auto k=kind==5?int(n%5):kind;node.aggregation=kinds[k];
    if(k<2)continue;
    for(Index slot=0;slot<f.graph.source_counts[n];++slot)
      f.model.nodes[n].extra[(k==2?"agg_mass_":"agg_logit_")+std::to_string(slot)]=
        slot==0?f.model.input_scale[0]:at::full({},.125f*float(slot)-.25f,at::kFloat);
  }
  f.graph.compile();f.initial.identity=f.graph.identity;
}
test::Fixture graph_fixture(int kind) {
  auto f=test::retained_fixture(kind%2?0:3,0,3);coefficients(f,kind);return f;
}
test::Fixture exclusive(int kind,Index width) {
  test::Fixture f;auto& g=f.graph;g.nodes={{0}};
  g.nodes[0].memory="ema";g.nodes[0].full="tanh";
  g.regions={{1,true,false,"proposal","count-v1"}};g.inputs={0,0,0,0};g.outputs={0};g.compile();
  g.source_domain->input={2,0,1,0};g.compile();
  NodeWeights w{at::zeros({width},at::kFloat),at::eye(width,at::kFloat)*.125f,
    at::full({width},.03125f,at::kFloat),at::full({width},.125f,at::kFloat)};
  f.model.nodes={w};auto shared=at::full({},.5f,at::kFloat);
  f.model.input_scale={shared,shared,at::full({},.75f,at::kFloat),shared};
  f.model.output_scale={at::full({},.25f,at::kFloat)};
  f.initial.batch_size=1;f.initial.states[{0,0}]={at::full({width},.015625f,at::kFloat),-1,0};
  coefficients(f,kind);
  // Two physical alternatives carry one logical source at different times;
  // another logical source is absent at t0 and present-zero at t1.
  f.input={{0,1,0,0,at::full({width},.25f,at::kFloat)},
    {0,2,0,0,at::full({width},.125f,at::kFloat)},
    {0,3,0,1,at::full({width},-.125f,at::kFloat)},
    {0,2,1,1,at::full({width},.375f,at::kFloat)},
    {0,0,0,1,at::zeros({width},at::kFloat)}};
  return f;
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);
    if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("Aggregate training requires explicit NPU FP32");
    auto device=portable_torch::resolve_device(args);
    at::set_num_threads(1);at::set_num_interop_threads(1);at::NoGradGuard guard;int cases=0;
    for(int kind=1;kind<=5;++kind)for(bool prefill:{false,true})
    for(auto optimizer:{ResidentOptimizerKind::sgd,ResidentOptimizerKind::adamw})for(auto dtype:{at::kFloat,at::kDouble}) {
      try{test::train_trajectory(device,graph_fixture(kind),prefill,optimizer,dtype,1e-5);++cases;}
      catch(...){std::cerr<<"Aggregate training kind="<<kind<<" prefill="<<prefill<<" reference="<<dtype<<'\n';throw;}
    }
    for(int kind=1;kind<5;++kind)for(bool prefill:{false,true})for(auto dtype:{at::kFloat,at::kDouble}) {
      try{test::train_trajectory(device,exclusive(kind,3),prefill,ResidentOptimizerKind::adamw,dtype,1e-5);++cases;}
      catch(...){std::cerr<<"Aggregate exclusive alias kind="<<kind<<" prefill="<<prefill<<" reference="<<dtype<<'\n';throw;}
    }
    test::train_trajectory(device,exclusive(4,257),true,ResidentOptimizerKind::adamw,at::kFloat,1e-5);++cases;
    std::cout<<"resident-aggregate-training: passed trajectories="<<cases<<" windows="<<cases*16<<" updates="<<cases*4
      <<" CPU=FP32_FP64 shared_owners=true exclusive_sources=true checkpoint_resume=true adam_epsilon=1e-5\n";
    runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
