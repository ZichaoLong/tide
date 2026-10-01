#include "execution.h"
#include "../../cpp/bench/streaming.h"
#include <ATen/Parallel.h>
#include <iostream>

int main() {
  using namespace accelerator_scale;
  try {
    at::NoGradGuard guard;at::set_num_threads(1);
    pdg_scale::Topology topology{4,2,2,1,1,2,{}};
    for(auto [s,t]:std::vector<std::pair<Index,Index>>{{0,0},{4,4},{0,4},{4,0}}) {
      for(Index i=0;i<4;++i)topology.edges.push_back({s,t+i,1});
      for(Index i=1;i<4;++i)topology.edges.push_back({s+i,t,1});
    }
    for(auto dtype:{at::kDouble,at::kFloat})for(const std::string controls:{"cpu","model"}) {
      pdg_scale::Config c;c.runtime.dtype=dtype;c.memory="add";c.width=8;c.vocab=17;
      portable_torch::seed_runtime(at::Device(at::kCPU),7);
      auto f=pdg_scale::fixture(c,topology);Scoring scoring{"cpu",controls,dtype};
      configure_scoring(f,scoring);
      auto placement=place(f,at::Device(at::kCPU),1,"locality",true);placement.scoring=scoring;
      Options options;options.trace=true;
      Streaming reference(f.graph,f.model,options);Resident actual(f.graph,f.model,options,1,placement);
      Continuation q;q.identity=reference.graph().identity;q.batch_size=1;
      for(Index token=0;token<2;++token) {
        std::vector<External> input{{0,0,token,token*3,f.embedding[token]}};
        auto expected=reference.run(q,input,(token+1)*3,(token+1)*3);q=expected.continuation;
        auto got=actual.advance(input,(token+1)*3,(token+1)*3);
        Result result;result.trace=got.trace;result.messages=got.messages;result.outputs=got.outputs;result.continuation=actual.snapshot();
        tide_bench::compare(result,expected,true,dtype,std::nullopt,dtype==at::kDouble?1e-8:1e-5,dtype==at::kDouble?1e-10:1e-6);
        if(result.trace.empty())throw std::runtime_error("empty resident control dtype fixture");
        for(const auto& e:result.trace)if(e.control.scalar_type()!=dtype)
          throw std::runtime_error("resident control precision changed from independent reference");
      }
    }
    std::cout<<"PASS resident CPU FP64/FP32 controls and two-window observables\n";
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
