// Independent installed C++ master-policy adapter, including aliases/None/zero.
#include "eager_optimizer.h"
#include <ATen/Parallel.h>
#include <iostream>
#include <limits>

using namespace tide_flow;
void probe(int argc,char** argv) {
    if(argc!=3)throw std::invalid_argument("expected explicit device and optimizer");
    portable_torch::RuntimeOptions options;options.device_spec=argv[1];options.dtype=at::kHalf;
    options.allow_npu_float16=true;auto device=portable_torch::resolve_device(options);
    at::set_num_threads(1);at::set_num_interop_threads(1);
    tide::ParameterRegistry parameters;
    for(const auto& name:{"active","zero","unused","intermittent"})
      parameters.add(name,at::tensor({1.,-.5},at::kFloat).to(device,at::kHalf).set_requires_grad(true));
    parameters.add("shared",parameters.value("active"));
    tide::OptimizerGroup group;group.lr=.01;group.weight_decay=.1;group.momentum=.25;group.eps=1e-6;
    EagerOptimizer optimizer(parameters,argv[2],true,128,group);
    if(optimizer.inner().registry().alias_partitions()!=parameters.alias_partitions())
      throw std::runtime_error("master aliases changed");
    for(Index step=0;step<3;++step) {
      optimizer.zero_grad();
      for(int part=0;part<2;++part) {
        auto loss=parameters.value("active").to(at::kFloat).sum()*((step+1)*.03125)
          +parameters.value("zero").to(at::kFloat).sum()*0;
        if(step!=1)loss=loss+parameters.value("intermittent").to(at::kFloat).sum()*.0625;
        optimizer.backward(loss*.5);
      }
      optimizer.step();std::cout<<"{\"step\":"<<step<<",\"owners\":{";bool first=true;
      for(const auto& owner:parameters.owners()) {
        if(!first)std::cout<<',';first=false;const auto master=optimizer.inner().registry().value(owner.canonical);
        std::cout<<quoted(owner.canonical)<<":{\"payload\":";tensor_json(std::cout,owner.value);
        std::cout<<",\"gradient\":";tensor_json(std::cout,owner.value.grad());
        std::cout<<",\"master\":";tensor_json(std::cout,master);
        const auto it=optimizer.inner().state().find(owner.canonical);
        std::cout<<",\"slot\":";
        if(it==optimizer.inner().state().end())std::cout<<"null";
        else {const auto& s=it->second;std::cout<<"{\"step\":"<<s.step<<",\"first\":";
          tensor_json(std::cout,s.momentum_buffer.defined()?s.momentum_buffer:s.exp_avg);
          std::cout<<",\"second\":";tensor_json(std::cout,s.exp_avg_sq);std::cout<<'}';}
        std::cout<<'}';
      }
      std::cout<<"}}\n";
    }
    std::vector<Tensor> before;
    for(const auto& owner:optimizer.inner().registry().owners())before.push_back(owner.value.detach().clone());
    optimizer.zero_grad();optimizer.backward(parameters.value("active").to(at::kFloat).sum()*std::numeric_limits<float>::infinity());
    bool rejected=false;try{optimizer.step();}catch(const std::runtime_error& e){rejected=std::string(e.what()).find("nonfinite")!=std::string::npos;}
    size_t i=0;for(const auto& owner:optimizer.inner().registry().owners())
      if(!at::equal(before[i++],owner.value))throw std::runtime_error("nonfinite gradient changed masters");
    if(!rejected)throw std::runtime_error("nonfinite gradient accepted");
    std::cout<<"{\"nonfinite_rejected\":true,\"aliases_preserved\":true}\n";
    portable_torch::synchronize(device);
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {probe(argc,argv);runtime.close();return 0;}
  catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}
}
