#include "eager.h"
#include <ATen/core/grad_mode.h>
#include "tide/greedy.h"
#include "tide/stream.h"
#include "tide/ownership.h"
#include "tide/transfer.h"
#include "tide/graph_checkpoint.h"
#include <torch/csrc/autograd/autograd.h>
#include "../device_online/weighted_precision_reference.h"
#include "../../cpp/bench/comparison.h"
#include <cmath>
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
             ExecutionPlacement placement,Index workers,std::vector<at::Device> peers,bool storage,bool return_packed,
             std::vector<at::Device> owners,std::string returns)
  :data(fixture(j,dtype)),training(train),prefill(batch),storage_reference(storage),return_packed(return_packed),
   return_mode(std::move(returns)),loss_scale(j.value("__task",Json::object()).value("loss_scale",.001)){
  if(workers<1)throw std::invalid_argument("workers must be positive");
  if(storage){
    if(dtype!=at::kHalf||!device.is_cpu()||!scalar)throw std::invalid_argument("half storage reference requires CPU scalar execution");
    data.model=tide::device_online::weighted_precision_check::reference_model(data.model);
    std::map<const void*,Tensor> copies;
    for(auto& x:data.inputs){const auto key=x.value.unsafeGetTensorImpl();
      if(!copies.count(key))copies[key]=x.value.to(at::kFloat);x.value=copies.at(key);}
  }
  if(peers.empty())peers.push_back(device);
  if(owners.empty())for(const auto& node:data.graph.nodes)owners.push_back(peers[node.region%peers.size()]);
  data.model=place_payloads(data.graph,data.model,owners);
  placed=place_model(data.graph,data.model,placement);registry=data.model.parameters();
  state.identity=data.graph.identity;state.batch_size=data.samples;
  options.workers=workers;options.packed=!scalar;
  options.full_autograd=options.aggregate_autograd=scalar?"replay":"batched";
  OptimizerGroup group;for(const auto& p:registry.owners())group.parameters.push_back(p.canonical);
  const auto task=j.value("__task",Json::object());
  group.lr=task.value("lr",.001);group.momentum=task.value("momentum",.3);group.weight_decay=task.value("weight_decay",.01);
  if(training)optimizer=std::make_unique<SGD>(registry,std::vector<OptimizerGroup>{group},OptimizerPrecision{dtype==at::kHalf&&!storage,1.});
  if(training&&storage)for(const auto& p:registry.owners())storage_masters[p.canonical]=p.value.detach().clone();
}
std::vector<External> Eager::placed_inputs(std::vector<External> xs,std::map<std::string,Tensor>* leaves)const{
  std::map<std::pair<const void*,std::string>,Tensor> copies;
  std::map<const void*,std::string> names;
  for(auto& x:xs){const auto device=data.model.nodes[data.graph.inputs[x.port]].bias.device();
    if(leaves&&!names.count(x.value.unsafeGetTensorImpl())){
      const auto name="input."+std::to_string(x.batch)+"."+std::to_string(x.port)+"."+std::to_string(x.position)+"."+std::to_string(x.time);
      names[x.value.unsafeGetTensorImpl()]=name;leaves->emplace(name,x.value);}
    const auto key=std::make_pair(x.value.unsafeGetTensorImpl(),device.str());
    if(!copies.count(key))copies[key]=x.value.device()==device?x.value:copy_rows({x.value},device).front();
    x.value=copies.at(key);}
  return xs;
}
Result Eager::forward(const Continuation& q,const std::vector<External>& xs,Index stop)const{
  return prefill?Greedy(data.graph,placed,options).run(q,xs,stop,stop)
                :Streaming(data.graph,placed,options).run(q,xs,stop,stop);
}
Step Eager::step(Index index){
  if(index<0||index!=state.cut/data.span||state.cut%data.span)throw std::invalid_argument("window index disagrees with live continuation cut");
  at::AutoGradMode grad(training);if(optimizer)optimizer->zero_grad();
  auto r=forward(state,placed_inputs(inputs(data,index)),(index+1)*data.span);
  auto value=!training&&return_mode=="metrics"?at::zeros({}):loss(r,!training,loss_scale);
  if(training){optimizer->backward(value);bool connected=false;std::map<std::string,Tensor> gradients;
    for(const auto& p:registry.owners())if(p.value.grad().defined()){
      connected=true;gradients[p.canonical]=p.value.grad();}
    tide_bench::check_finite(gradients,"parameter gradients");
    if(!connected)throw std::runtime_error("benchmark has no connected parameter gradient");
    if(storage_reference){
      at::NoGradGuard ng;
      for(const auto& p:registry.owners())p.value.copy_(storage_masters.at(p.canonical));
      optimizer->step();
      for(const auto& p:registry.owners()){
        storage_masters.at(p.canonical).copy_(p.value);p.value.copy_(p.value.to(at::kHalf).to(at::kFloat));}
    }else optimizer->step();}
  const auto numeric=value.cpu().item<double>();
  if(!std::isfinite(numeric))throw tide_bench::ComparisonFailure({tide_bench::DifferenceKind::nonfinite,"loss","nonfinite objective"});
  state=detach_live(r.continuation);
  Json observation;
  if(return_mode!="complete")observation=work(r);
  auto output=projected(r,return_mode,return_packed);
  tide_bench::check_finite(return_mode=="complete"?output:r);
  if(training){std::map<std::string,Tensor> fields;for(const auto& p:registry.owners())fields["parameter."+p.canonical]=p.value;
    for(const auto& [name,s]:optimizer->state())fields["momentum."+name]=s.momentum_buffer;
    for(const auto& [name,x]:optimizer->masters())fields["master."+name]=x;
    for(const auto& [name,x]:storage_masters)fields["storage_master."+name]=x;
    tide_bench::check_finite(fields,"updated training owners");}
  return {std::move(output),numeric,std::move(observation)};
}
std::map<std::string,Tensor> Eager::parameters()const{
  std::map<std::string,Tensor> out;for(const auto& p:registry.owners())out[p.canonical]=cpu(storage_reference?p.value.to(at::kHalf):p.value);return out;
}
TrainingState Eager::training_state()const{
  TrainingState out;if(!training)return out;
  for(const auto& p:registry.owners()){
    const auto g=p.value.grad();out.gradients[p.canonical]=cpu(g.defined()&&g.scalar_type()==at::kHalf?g.to(at::kFloat):g);
    const auto found=optimizer->state().find(p.canonical);
    out.momentum[p.canonical]=found==optimizer->state().end()?Tensor{}:cpu(found->second.momentum_buffer);
  }
  for(const auto& [name,x]:optimizer->masters())out.masters[name]=cpu(x);
  if(storage_reference)for(const auto& [name,x]:storage_masters)out.masters[name]=cpu(x);
  return out;
}
Continuation Eager::checkpoint()const{return detached(state);}
void Eager::save(const std::filesystem::path& p,int64_t limit)const{GraphCheckpoint::save(p,data.graph,data.model,state,optimizer.get(),limit);}
void Eager::restore(const std::filesystem::path& p,int64_t limit){state=GraphCheckpoint::load(p,data.graph,data.model,optimizer.get(),limit);}
}
