// Public installed headers only. Independent identity-VJP and detached resume.
// Optional physical relocation follows docs/resident-weighted-placement.md.
#include <tide/resident_weighted_training.h>
#include <portable_torch/runtime.hpp>
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include "fixture.h"
#include <iostream>
int main(int argc,char** argv){portable_torch::RuntimeSession runtime;
  try{bool placed=false;std::vector<char*> filtered{argv[0]};
    for(int i=1;i<argc;++i)if(std::string(argv[i])=="--node-placed")placed=true;else filtered.push_back(argv[i]);
    auto args=portable_torch::parse_cli(filtered.size(),filtered.data(),true);if(args.help){
      portable_torch::print_usage(std::cout,argv[0]);std::cout<<"  --node-placed  Use logical peer 1, then restore on coordinator 0\n";return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("weighted training consumer requires explicit accelerator FP32");
    auto device=portable_torch::resolve_device(args);at::set_num_threads(1);at::set_num_interop_threads(1);at::NoGradGuard guard;
    if(placed&&device.index()!=0)throw std::invalid_argument("placed consumer requires coordinator 0 and peer 1");
    using namespace tide;using namespace tide::device_online;auto fixture=weighted_consumer_fixture();
    fixture.model.nodes[0].weight.set_requires_grad(true); // Legacy weight has no role in this binding.
    WeightedTrainingLimits limits;limits.forward.queue=4;limits.forward.cache=4;limits.forward.stages=4;
    std::vector<at::Device> placement{at::Device(device.type(),placed?1:device.index())};
    auto owner=placed?std::make_unique<WeightedTrainingSession>(fixture.graph,fixture.model,fixture.initial,device,ResidentOptimizerKind::adamw,std::vector<OptimizerGroup>{},limits,placement):
      std::make_unique<WeightedTrainingSession>(fixture.graph,fixture.model,fixture.initial,device,ResidentOptimizerKind::adamw,std::vector<OptimizerGroup>{},limits);
    for(int step=0;step<2;++step){auto x=step?at::tensor({.4f,-3.f}):at::tensor({.6f,2.f});
      if(owner->node_devices()!=placement)throw std::runtime_error("installed public node placement differs");
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
      owner->close();if(placed)placement[0]=at::Device(device.type(),step%2);
      owner=placed?std::make_unique<WeightedTrainingSession>(fixture.graph,fixture.model,c,device,limits,placement):
        std::make_unique<WeightedTrainingSession>(fixture.graph,fixture.model,c,device,limits);
    }
    owner->close();std::cout<<"weighted-training-consumer: passed hand_identity_vjp=true none_optimizer=true exact_cut_resume=true node_placed="<<placed<<"\n";return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
