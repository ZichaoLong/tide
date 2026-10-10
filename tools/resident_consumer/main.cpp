// Installed-package client: no private backend header or repository include.
#include <tide/resident.h>
#include <tide/resident_training.h>
#include <tide/stream.h>
#include <portable_torch/runtime.hpp>
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <iostream>
#include <stdexcept>

int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    int cards=1;std::vector<char*> forwarded{argv[0]};
    for(int i=1;i<argc;++i) {
      const std::string argument=argv[i];
      if(argument.rfind("--training-devices=",0)==0) {
        const auto value=argument.substr(19);size_t used=0;cards=std::stoi(value,&used);
        if(used!=value.size()||cards<1||cards>16)throw std::invalid_argument("training-devices must be 1..16");
      } else forwarded.push_back(argv[i]);
    }
    auto args=portable_torch::parse_cli(forwarded.size(),forwarded.data(),true);
    if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("resident consumer requires explicit accelerator FP32");
    auto device=portable_torch::resolve_device(args);
    at::set_num_threads(1);at::set_num_interop_threads(1);
    {
      at::NoGradGuard guard;
      tide::Graph graph;graph.nodes={{0},{1}};graph.regions={{1},{1}};
      graph.edges={{0,1,1},{1,0,2}};graph.inputs={0};graph.outputs={1};graph.compile();
      tide::Model model;
      for(int n=0;n<2;++n)model.nodes.push_back({at::zeros({2}),at::eye(2),at::zeros({2}),at::ones({2})});
      model.input_scale={at::ones({})};model.output_scale={at::ones({})};
      model.edge_scale={at::full({},.25),at::full({},.25)};model.agg_scale={at::ones({}),at::ones({})};
      tide::Continuation q;q.identity=graph.identity;q.batch_size=1;
      tide::Streaming reference(graph,model,{});
      tide::ResidentLimits limits;limits.queue=32;limits.arrivals=32;limits.outputs=32;limits.trace=128;
      tide::ResidentSession session(graph,model,q,device,limits);
      for(tide::Index step=0;step<3;++step) {
        const auto cut=3*(step+1);
        std::vector<tide::External> input={{0,0,step,step*3,at::tensor({.5f,-.25f})}};
        const auto expected=reference.run(q,input,cut,cut);q=expected.continuation;
        const auto view=session.advance(input,cut,cut);
        if(view.values.device()!=device)throw std::runtime_error("output escaped NPU");
        const auto got=session.result();
        if(got.outputs.size()!=expected.outputs.size()||got.continuation.pending.size()!=q.pending.size())
          throw std::runtime_error("installed consumer discrete mismatch");
        for(size_t i=0;i<got.outputs.size();++i) {
          const auto& a=got.outputs[i];const auto& b=expected.outputs[i];
          if(a.batch!=b.batch||a.time!=b.time||a.port!=b.port||!at::allclose(a.value,b.value,1e-5,1e-6))
            throw std::runtime_error("installed consumer output mismatch");
        }
      }
      session.close();
      for(const auto& owner:model.parameters(false).owners())owner.value.set_requires_grad(true);
      tide::ResidentTrainingLimits training_limits;training_limits.forward=limits;
      if(cards>1) {
        for(int i=0;i<cards;++i)training_limits.placement.devices.emplace_back(device.type(),device.index()+i);
        training_limits.forward.workspace_bytes=512*1024*1024;
        training_limits.backward_bytes=tide::Index(4)*1024*1024*1024;
      }
      tide::ResidentTrainingSession training(graph,model,q,device,tide::ResidentOptimizerKind::adamw,{},training_limits);
      auto before=training.checkpoint();std::vector<tide::ResidentCotangents> roots;
      for(tide::Index step=3;step<5;++step) {
        const auto stop=3*(step+1);
        auto window=training.advance({{0,0,step,step*3,at::tensor({.5f,-.25f})}},stop,stop);
        tide::ResidentCotangents cot;cot.token=window.token;cot.outputs=at::ones_like(window.outputs.values);
        cot.outputs_connected=window.outputs.valid.clone();roots.push_back(cot);
      }
      auto gradient=training.backward(roots);
      if(cards==1) {
        if(gradient.values.device()!=device)throw std::runtime_error("installed gradient left device");
      } else {
        if(gradient.values.defined()||gradient.parameter_shards.size()!=size_t(cards))throw std::runtime_error("installed gradient shards missing");
        for(size_t i=0;i<gradient.parameter_shards.size();++i)
          if(gradient.parameter_shards[i].values.device()!=training_limits.placement.devices[i])throw std::runtime_error("gradient left owner");
      }
      if(!training.step().applied)throw std::runtime_error("installed training step failed");
      auto checkpoint=training.checkpoint();training.close();
      if(at::equal(before.state.values,checkpoint.state.values))throw std::runtime_error("installed training made no parameter update");
      training_limits.placement.policy="memory";
      tide::ResidentTrainingSession restored(graph,model,checkpoint,device,training_limits);
      restored.advance({{0,0,5,15,at::tensor({.5f,-.25f})}},18,18);restored.detach();restored.close();
    }
    runtime.close();std::cout<<"installed-resident: passed windows=3 feedback=true training_windows=3 retained_backward=true optimizer_restore=true training_devices="<<cards<<'\n';return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
