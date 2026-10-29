#include "eager.h"
#include <ATen/core/grad_mode.h>
#include "tide/greedy.h"
#include "tide/stream.h"
#include "tide/ownership.h"
#include "tide/graph_checkpoint.h"
#include <torch/csrc/autograd/autograd.h>
namespace weighted_benchmark {
namespace {
Continuation detach_live(Continuation q){
  for(auto& [_,s]:q.states){s.value=s.value.detach();for(auto& [__,x]:s.slots)x=x.detach();}
  for(auto& [_,h]:q.history)for(auto& [__,x]:h.tensors)x=x.detach();
  for(auto& a:q.pending)a.value=a.value.detach();return q;
}
Tensor cpu(const Tensor& x){return x.defined()?x.detach().cpu().clone():Tensor{};}
}
Eager::Eager(const Json& j,bool train,bool batch,bool scalar,at::ScalarType dtype,at::Device device,
             ExecutionPlacement placement,Index workers,std::vector<at::Device> peers)
  :data(fixture(j,dtype)),training(train),prefill(batch){
  if(workers<1)throw std::invalid_argument("workers must be positive");
  if(peers.empty())peers.push_back(device);
  std::vector<at::Device> owners;for(const auto& node:data.graph.nodes)owners.push_back(peers[node.region%peers.size()]);
  data.model=place_payloads(data.graph,data.model,owners);
  placed=place_model(data.graph,data.model,placement);registry=data.model.parameters();
  state.identity=data.graph.identity;state.batch_size=data.samples;
  options.workers=workers;options.packed=!scalar;
  options.full_autograd=options.aggregate_autograd=scalar?"replay":"batched";
  OptimizerGroup group;for(const auto& p:registry.owners())group.parameters.push_back(p.canonical);
  group.lr=.001;group.momentum=.3;group.weight_decay=.01;
  optimizer=std::make_unique<SGD>(registry,std::vector<OptimizerGroup>{group},OptimizerPrecision{dtype==at::kHalf,1.});
}
Step Eager::step(Index index){
  at::AutoGradMode grad(training);optimizer->zero_grad();auto xs=inputs(data,index);
  std::map<std::pair<const void*,std::string>,Tensor> copies;
  for(auto& x:xs){const auto device=data.model.nodes[data.graph.inputs[x.port]].bias.device();
    const auto key=std::make_pair(x.value.unsafeGetTensorImpl(),device.str());
    if(!copies.count(key))copies[key]=x.value.to(device);
    x.value=copies.at(key);}
  auto r=prefill?Greedy(data.graph,placed,options).run(state,xs,(index+1)*data.span,(index+1)*data.span)
                :Streaming(data.graph,placed,options).run(state,xs,(index+1)*data.span,(index+1)*data.span);
  auto value=loss(r);
  if(training){optimizer->backward(value);bool connected=false;
    for(const auto& p:registry.owners())if(p.value.grad().defined()){
      connected=true;if(!at::isfinite(p.value.grad()).all().item<bool>())throw std::runtime_error("nonfinite benchmark gradient");}
    if(!connected)throw std::runtime_error("benchmark has no connected parameter gradient");optimizer->step();}
  state=detach_live(r.continuation);return {detached(r),value.cpu().item<double>()};
}
std::map<std::string,Tensor> Eager::parameters()const{
  std::map<std::string,Tensor> out;for(const auto& p:registry.owners())out[p.canonical]=cpu(p.value);return out;
}
TrainingState Eager::training_state()const{
  TrainingState out;
  for(const auto& p:registry.owners()){
    const auto g=p.value.grad();out.gradients[p.canonical]=cpu(g.defined()&&g.scalar_type()==at::kHalf?g.to(at::kFloat):g);
    const auto found=optimizer->state().find(p.canonical);
    out.momentum[p.canonical]=found==optimizer->state().end()?Tensor{}:cpu(found->second.momentum_buffer);
  }
  for(const auto& [name,x]:optimizer->masters())out.masters[name]=cpu(x);
  return out;
}
void Eager::save(const std::filesystem::path& p)const{GraphCheckpoint::save(p,data.graph,data.model,state,optimizer.get());}
void Eager::restore(const std::filesystem::path& p){state=GraphCheckpoint::load(p,data.graph,data.model,optimizer.get());}
}
