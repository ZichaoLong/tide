// Public installed headers only. Independent identity-VJP and detached resume.
#include <tide/resident_weighted_training.h>
#include <portable_torch/runtime.hpp>
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include "fixture.h"
#include <iostream>
int main(int argc,char** argv){portable_torch::RuntimeSession runtime;
  try{auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("weighted training consumer requires explicit accelerator FP32");
    auto device=portable_torch::resolve_device(args);at::set_num_threads(1);at::set_num_interop_threads(1);at::NoGradGuard guard;
    using namespace tide;using namespace tide::device_online;auto fixture=weighted_consumer_fixture();
    fixture.model.nodes[0].weight.set_requires_grad(true); // Legacy weight has no role in this binding.
    WeightedTrainingLimits limits;limits.forward.queue=4;limits.forward.cache=4;limits.forward.stages=4;
    auto owner=std::make_unique<WeightedTrainingSession>(fixture.graph,fixture.model,fixture.initial,device,ResidentOptimizerKind::adamw,std::vector<OptimizerGroup>{},limits);
    for(int step=0;step<2;++step){auto x=step?at::tensor({.4f,-3.f}):at::tensor({.6f,2.f});
      auto w=owner->advance({{0,0,step,2*step,x}},2*step+1);const auto& output=w.fields.at("outputs");
      auto bar=at::full_like(output.values,NAN);auto mask=output.present.unsqueeze(-1).expand_as(bar);
      auto weights=at::tensor({1.f,2.f}).to(device).expand_as(bar);bar=at::where(mask,weights,bar);
      WeightedCotangents root;root.token=w.token;root.fields["outputs"]={bar,output.present};
      auto g=owner->backward({root});const auto& boundary=g.boundaries.at(0);auto valid=boundary.valid.cpu();
      int count=0;for(int64_t i=0;i<valid.numel();++i)if(valid[i].item<bool>()){
        if(!boundary.connected[i].item<bool>()||!at::allclose(boundary.values[i].cpu(),at::tensor({1.f,2.f}),1e-5,1e-6))
          throw std::runtime_error("installed public weighted VJP differs from identity");++count;}
      if(count!=1||g.parameters.connected.any().item<bool>()||!owner->step().applied)
        throw std::runtime_error("installed weighted None/update contract differs");
      auto c=owner->checkpoint();if(c.state.steps.ne(0).any().item<bool>())throw std::runtime_error("unused owner optimizer counter advanced");
      owner->close();owner=std::make_unique<WeightedTrainingSession>(fixture.graph,fixture.model,c,device,limits);
    }
    owner->close();std::cout<<"weighted-training-consumer: passed hand_identity_vjp=true none_optimizer=true exact_cut_resume=true\n";return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
