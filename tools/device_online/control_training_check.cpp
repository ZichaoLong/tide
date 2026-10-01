#include "training_test.h"
#include "portable_torch/runtime.hpp"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <iostream>

namespace {
using namespace tide;using namespace tide::device_online;
test::Fixture fixture(int variant,Index width=3) {
  auto f=test::retained_fixture(variant%4,variant==4?1:0,width);
  for(auto& r:f.graph.regions)r.read_mode=variant%3==0?"content":variant%3==1?"old":"proposal";
  for(size_t n=0;n<f.graph.nodes.size();++n)if(!f.graph.nodes[n].identity) {
    auto& node=f.graph.nodes[n];
    if(variant>=3)node.readout="norm-fp32-v1";
    if(variant==5) {
      node.aggregation="all_softmax";
      for(Index slot=0;slot<f.graph.source_counts[n];++slot)
        f.model.nodes[n].extra["agg_logit_"+std::to_string(slot)]=slot==0?f.model.input_scale[0]:at::full({},.125f*float(slot),at::kFloat);
    }
  }
  f.graph.compile();f.initial.identity=f.graph.identity;return f;
}
void refusals(at::Device device) {
  auto f=fixture(0);ResidentTrainingLimits l;l.forward.trace=512;l.forward.mode="hst";l.forward.zeta=.375;
  ResidentTrainingSession s(f.graph,f.model,f.initial,device,ResidentOptimizerKind::sgd,{},l);
  auto checkpoint=s.checkpoint();s.close();
  auto bad=l;bad.forward.mode="softp";
  test::train_reject([&]{ResidentTrainingSession r(f.graph,f.model,checkpoint,device,bad);},"checkpoint changed Emit mode");
  bad=l;bad.forward.zeta=0.;
  test::train_reject([&]{ResidentTrainingSession r(f.graph,f.model,checkpoint,device,bad);},"checkpoint changed surrogate zeta");
  bad=l;bad.forward.mode="unknown";
  test::train_reject([&]{ResidentTrainingSession r(f.graph,f.model,f.initial,device,ResidentOptimizerKind::sgd,{},bad);},"unknown Emit mode accepted");
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("control training requires explicit NPU FP32");
    auto device=portable_torch::resolve_device(args);at::set_num_threads(1);at::set_num_interop_threads(1);at::NoGradGuard guard;int cases=0;
    for(int variant=0;variant<6;++variant)for(auto mode:{"hst","softp"})for(bool prefill:{false,true})
    for(auto optimizer:{ResidentOptimizerKind::sgd,ResidentOptimizerKind::adamw})for(auto dtype:{at::kFloat,at::kDouble}) {
      Options options;options.mode=mode;options.zeta=.375;
      try{test::train_trajectory(device,fixture(variant),prefill,optimizer,dtype,1e-5,false,options);++cases;}
      catch(...){std::cerr<<"control training variant="<<variant<<" mode="<<mode<<" prefill="<<prefill<<" reference="<<dtype<<'\n';throw;}
    }
    for(auto mode:{"hst","softp"}) {
      Options options;options.mode=mode;options.zeta=0.;
      test::train_trajectory(device,fixture(0,257),true,ResidentOptimizerKind::sgd,at::kFloat,1e-5,false,options);++cases;
    }
    refusals(device);
    std::cout<<"resident-control-training: passed trajectories="<<cases<<" windows="<<cases*16<<" updates="<<cases*4
      <<" CPU=FP32_FP64 read_modes=content_old_proposal norm_zero_local=true shared_owners=true checkpoint_resume=true adam_epsilon=1e-5\n";
    runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
