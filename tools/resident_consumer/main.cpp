// Installed-package client: no private backend header or repository include.
#include <tide/resident.h>
#include <tide/stream.h>
#include <portable_torch/runtime.hpp>
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <iostream>
#include <stdexcept>

int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);
    if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("resident consumer requires explicit NPU FP32");
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
    }
    runtime.close();std::cout<<"installed-resident: passed windows=3 feedback=true\n";return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
