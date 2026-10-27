// Public installed headers only. A one-node identity Full is a hand oracle:
// Agg=(m,z), regional_total with gamma=1 returns c=m, and G=z, never m*z.
#include <tide/resident_weighted.h>
#include <tide/weighted.h>
#include <portable_torch/runtime.hpp>
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <iostream>
int main(int argc,char** argv){portable_torch::RuntimeSession runtime;
  try{
    auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("weighted consumer requires explicit accelerator FP32");
    auto device=portable_torch::resolve_device(args);at::set_num_threads(1);at::set_num_interop_threads(1);at::NoGradGuard guard;
    using namespace tide;Graph g;g.nodes={{0}};g.regions={{1,true,false,"proposal","weighted-consumer"}};g.inputs=g.outputs={0};
    auto& n=g.nodes[0];n.memory=n.readout=n.emission=n.aggregation="weighted-consumer";g.compile();
    weighted::KVConfig c;c.width=c.summary_width=c.adapter_rank=c.heads=c.ffn_width=1;c.alpha_att=c.alpha_ff=0;
    weighted::RegionalConfig r;r.hidden_width=1;r.region_control_mode="fixed";r.region_evidence.reset();r.total_fraction=.5;
    NodeWeights w{at::zeros({2}),at::zeros({2,2}),at::zeros({2}),at::zeros({2})};
    w.extra={{"kernel.base",at::zeros({2,1})},{"kernel.adapter_u",at::zeros({2,1})},{"kernel.adapter_v",at::zeros({1,1})},
      {"kernel.summary",at::ones({1,1})},{"read_program.score.matrix",at::zeros({1,8})},{"read_program.score.hidden_bias",at::zeros({1})},
      {"read_program.score.vector",at::zeros({1})},{"read_program.score.bias",at::zeros({})}};
    for(const auto& name:{"query","out","gate","up","down"})w.extra["full_program.content."+std::string(name)]=at::zeros({1,1});
    weighted::configure_node(w,c,r,SourceClasses{{},{0}},0);Model m;m.nodes={w};m.regions.resize(1);weighted::configure_region(m.regions[0],r);
    m.input_scale=m.output_scale={at::ones({})};Continuation q;q.identity=g.identity;q.batch_size=1;
    device_online::WeightedFlowLimits limits;limits.queue=4;limits.cache=4;limits.stages=4;
    for(int step=0;step<2;++step){
      device_online::WeightedFlow owner(g,m,q,device,limits);auto value=step?at::tensor({.4f,-3.f}):at::tensor({.6f,2.f});
      auto result=owner.advance({{0,0,step,2*step,value}},2*step+1);
      if(result.outputs.size()!=1||result.trace.size()!=1||!result.messages.empty()||!result.continuation.pending.empty()
          ||!at::allclose(result.outputs[0].value,value,1e-5,1e-6)||result.continuation.states.at({0,0}).observations!=step+1)
        throw std::runtime_error("installed weighted consumer differs from hand identity/continuation");
      q=owner.checkpoint();owner.close();
    }
    std::cout<<"weighted-resident-consumer: passed hand_identity=true exact_continuation=true\n";return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
