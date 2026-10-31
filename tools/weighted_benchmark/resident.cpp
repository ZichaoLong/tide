#include "benchmark.h"
#include "resident_checkpoint.h"
#include "tide/resident_weighted_training.h"
#include "tide/parameters.h"
#include "tide/graph_checkpoint.h"
#include "portable_torch/runtime.hpp"
#include "../../cpp/bench/comparison.h"
#include <cmath>
#include <torch/csrc/autograd/autograd.h>
#include <set>
namespace weighted_benchmark {
using namespace tide::device_online;
namespace {
class Candidate final:public Resident {
  Json material;Fixture data;at::Device device;ResidentParameterGradient last_gradient;
  WeightedTrainingLimits limits;std::vector<at::Device> node_owners;at::ScalarType dtype;std::unique_ptr<WeightedFlow> inference;std::unique_ptr<WeightedTrainingSession> training;
  std::string return_mode;double loss_scale;Index current_cut=0;
 public:
  Candidate(const Json& j,at::Device d,bool train,bool prefill,at::ScalarType dtype,const std::vector<at::Device>& peers,
            std::vector<at::Device> owners,const std::string& returns):material(j),data(fixture(j,dtype)),device(d),dtype(dtype),
            return_mode(returns),loss_scale(j.value("__task",Json::object()).value("loss_scale",.001)){
    at::NoGradGuard ng;Continuation q;q.identity=data.graph.identity;q.batch_size=data.samples;
    WeightedTrainingLimits l;const auto& x=j.at("limits");
#define FIELD(name) l.forward.name=x.at(#name).get<decltype(l.forward.name)>()
    FIELD(queue);FIELD(stages);FIELD(cache);FIELD(tensor_bytes);FIELD(operator_bytes);FIELD(timeout_ms);
#undef FIELD
#define FIELD(name) l.name=x.at(#name).get<decltype(l.name)>()
    if(train){FIELD(retained_bytes);FIELD(backward_bytes);FIELD(optimizer_bytes);FIELD(total_bytes);}
#undef FIELD
    l.forward.prefill=prefill;
    if(owners.empty())for(const auto& node:data.graph.nodes)owners.push_back(peers.empty()?d:peers[node.region%peers.size()]);
    limits=l;node_owners=owners;
    if(train){OptimizerGroup group;for(const auto& p:data.model.parameters().owners())group.parameters.push_back(p.canonical);
      const auto task=j.value("__task",Json::object());
      group.lr=task.value("lr",.001);group.momentum=task.value("momentum",.3);group.weight_decay=task.value("weight_decay",.01);
      training=std::make_unique<WeightedTrainingSession>(data.graph,data.model,q,d,ResidentOptimizerKind::sgd,std::vector<OptimizerGroup>{group},l,owners);
    }else inference=std::make_unique<WeightedFlow>(data.graph,data.model,q,d,l.forward,owners);
  }
  Step step(Index index)override{
    if(index<0||index!=current_cut/data.span||current_cut%data.span)throw std::invalid_argument("window index disagrees with resident cut");
    at::NoGradGuard ng;auto xs=inputs(data,index);Result r;Tensor value;Json observation;
    if(training){auto w=training->advance(xs,(index+1)*data.span);const auto& f=w.fields.at("outputs");Tensor bar;
      {at::AutoGradMode grad(true);auto x=f.values.detach().to(at::kFloat).clone().set_requires_grad(true);
        auto numeric=x.narrow(-1,0,1)*x.narrow(-1,1,x.size(-1)-1);
        value=at::where(f.present.unsqueeze(-1),numeric.square(),at::zeros_like(numeric)).sum()*loss_scale;
        bar=torch::autograd::grad({value},{x})[0];}
      WeightedCotangents roots;roots.token=w.token;roots.fields["outputs"]={bar,f.present};
      if(return_mode=="complete")r=training->result();
      else if(return_mode=="outputs")r.outputs=training->outputs();
      auto gradients=training->backward({roots});last_gradient=gradients.parameters;if(!at::isfinite(gradients.parameters.values).all().item<bool>())throw std::runtime_error("nonfinite resident gradient");
      if(!training->step().applied)throw std::runtime_error("resident benchmark optimizer refused");
    }else {
      if(return_mode=="complete")r=inference->advance(xs,(index+1)*data.span);
      else {inference->advance_device(xs,(index+1)*data.span);if(return_mode=="outputs")r.outputs=inference->outputs();}
      value=return_mode=="metrics"?at::zeros({}):loss(r,true,loss_scale);
    }
    current_cut=(index+1)*data.span;
    if(return_mode!="complete"){
      r.continuation.identity=data.graph.identity;r.continuation.cut=current_cut;r.continuation.batch_size=data.samples;
      const auto stats=training?training->metrics():inference->metrics();
      observation={{"candidate_updates",stats.at("candidate_updates")},{"full_calls",stats.at("full_calls")},
        {"messages",stats.at("messages")},{"outputs",stats.at("outputs")},{"pending",stats.at("pending")},
        {"actual_stats",stats},{"boundary_activity",{{"state","not_checked"},{"reason","minimal device return; no complete state decode"}}}};
    }
    auto actual=detached(r);tide_bench::check_finite(actual);
    const auto number=value.cpu().item<double>();
    if(!std::isfinite(number))throw tide_bench::ComparisonFailure({tide_bench::DifferenceKind::nonfinite,"loss","nonfinite resident objective"});
    portable_torch::synchronize(device);return {std::move(actual),number,std::move(observation)};
  }
  std::map<std::string,Tensor> parameters()const override{
    at::NoGradGuard ng;std::map<std::string,Tensor> out;if(!training)return out;
    const auto c=training->checkpoint();for(const auto& p:data.model.parameters().owners())out[p.canonical]=c.parameters.at(p.canonical);
    return out;
  }
  TrainingState training_state()const override{
    at::NoGradGuard ng;TrainingState out;if(!training)return out;const auto c=training->checkpoint();
    const auto values=last_gradient.values.cpu(),connected=last_gradient.connected.cpu();
    const auto registry=data.model.parameters();
    for(const auto& p:registry.owners()){out.gradients[p.canonical]=Tensor{};out.momentum[p.canonical]=Tensor{};}
    for(size_t i=0;i<last_gradient.names.size();++i){const auto name=registry.canonical_name(last_gradient.names[i]);
      const auto parameter=registry.value(name);const auto offset=last_gradient.offsets[i];
      if(connected[i].item<bool>())out.gradients[name]=values.narrow(0,offset,parameter.numel()).reshape(parameter.sizes()).clone();}
    std::set<std::string> trainable(c.trainable.begin(),c.trainable.end());std::vector<std::string> names;
    for(const auto& [name,_]:c.parameters)if(trainable.count(name))names.push_back(name);
    if(names.size()!=c.offsets.size())throw std::runtime_error("resident optimizer owner offsets differ");
    for(size_t i=0;i<names.size();++i){const auto name=registry.canonical_name(names[i]);const auto p=registry.value(name);
      const auto master=c.state.values.narrow(0,c.offsets[i],p.numel()).reshape(p.sizes());
      if(!at::equal(master.to(p.scalar_type()),c.parameters.at(names[i])))throw std::runtime_error("resident master publication differs");
      if(p.scalar_type()==at::kHalf)out.masters[name]=master.clone();
      if(c.state.steps[i].item<Index>())out.momentum[name]=c.state.first.narrow(0,c.offsets[i],p.numel()).reshape(p.sizes()).clone();
    }
    return out;
  }
  void save(const std::filesystem::path& path,int64_t limit)const override{
    at::NoGradGuard ng;
    if(training)save_resident(path,fixture(material,dtype),training->checkpoint(),limit);
    else GraphCheckpoint::save(path,data.graph,data.model,inference->checkpoint(),nullptr,limit);
  }
  void restore(const std::filesystem::path& path,int64_t limit)override{
    at::NoGradGuard ng;
    if(training){
      training->close();training.reset();auto c=load_resident(path,data,limit);
      current_cut=c.continuation.cut;
      training=std::make_unique<WeightedTrainingSession>(data.graph,data.model,c,device,limits,node_owners);
    }else{
      inference->close();inference.reset();auto q=GraphCheckpoint::load(path,data.graph,data.model,nullptr,limit);
      current_cut=q.cut;
      inference=std::make_unique<WeightedFlow>(data.graph,data.model,q,device,limits.forward,node_owners);
    }
  }
  Continuation checkpoint()const override{
    at::NoGradGuard ng;return training?training->checkpoint().continuation:inference->checkpoint();
  }
  Index cut()const override{return current_cut;}
  void close()override{at::NoGradGuard ng;if(training)training->close();if(inference)inference->close();}
};
}
std::unique_ptr<Resident> resident(const Json& data,at::Device device,bool training,bool prefill,at::ScalarType dtype,
                                   const std::vector<at::Device>& peers,const std::vector<at::Device>& nodes,
                                   const std::string& returns){
  return std::make_unique<Candidate>(data,device,training,prefill,dtype,peers,nodes,returns);
}
}
