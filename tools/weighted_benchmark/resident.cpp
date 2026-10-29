#include "benchmark.h"
#include "resident_checkpoint.h"
#include "tide/resident_weighted_training.h"
#include "tide/parameters.h"
#include "portable_torch/runtime.hpp"
#include <torch/csrc/autograd/autograd.h>
#include <set>
namespace weighted_benchmark {
using namespace tide::device_online;
namespace {
class Candidate final:public Resident {
  Json material;Fixture data;at::Device device;ResidentParameterGradient last_gradient;
  WeightedTrainingLimits limits;std::vector<at::Device> node_owners;at::ScalarType dtype;std::unique_ptr<WeightedFlow> inference;std::unique_ptr<WeightedTrainingSession> training;
 public:
  Candidate(const Json& j,at::Device d,bool train,bool prefill,at::ScalarType dtype,const std::vector<at::Device>& peers):material(j),data(fixture(j,dtype)),device(d),dtype(dtype){
    at::NoGradGuard ng;Continuation q;q.identity=data.graph.identity;q.batch_size=data.samples;
    WeightedTrainingLimits l;const auto& x=j.at("limits");
#define FIELD(name) l.forward.name=x.at(#name).get<decltype(l.forward.name)>()
    FIELD(queue);FIELD(stages);FIELD(cache);FIELD(tensor_bytes);FIELD(operator_bytes);FIELD(timeout_ms);
#undef FIELD
#define FIELD(name) l.name=x.at(#name).get<decltype(l.name)>()
    FIELD(retained_bytes);FIELD(backward_bytes);FIELD(optimizer_bytes);FIELD(total_bytes);
#undef FIELD
    l.forward.prefill=prefill;
    std::vector<at::Device> owners;for(const auto& node:data.graph.nodes)owners.push_back(peers.empty()?d:peers[node.region%peers.size()]);
    limits=l;node_owners=owners;
    if(train){OptimizerGroup group;for(const auto& p:data.model.parameters().owners())group.parameters.push_back(p.canonical);
      group.lr=.001;group.momentum=.3;group.weight_decay=.01;
      training=std::make_unique<WeightedTrainingSession>(data.graph,data.model,q,d,ResidentOptimizerKind::sgd,std::vector<OptimizerGroup>{group},l,owners);
    }else inference=std::make_unique<WeightedFlow>(data.graph,data.model,q,d,l.forward,owners);
  }
  Step step(Index index)override{
    at::NoGradGuard ng;auto xs=inputs(data,index);Result r;Tensor value;
    if(training){auto w=training->advance(xs,(index+1)*data.span);const auto& f=w.fields.at("outputs");Tensor bar;
      {at::AutoGradMode grad(true);auto x=f.values.detach().to(at::kFloat).clone().set_requires_grad(true);
        auto numeric=x.narrow(-1,0,1)*x.narrow(-1,1,x.size(-1)-1);
        value=at::where(f.present.unsqueeze(-1),numeric.square(),at::zeros_like(numeric)).sum()*.001;
        bar=torch::autograd::grad({value},{x})[0];}
      WeightedCotangents roots;roots.token=w.token;roots.fields["outputs"]={bar,f.present};r=training->result();
      auto gradients=training->backward({roots});last_gradient=gradients.parameters;if(!at::isfinite(gradients.parameters.values).all().item<bool>())throw std::runtime_error("nonfinite resident gradient");
      if(!training->step().applied)throw std::runtime_error("resident benchmark optimizer refused");
    }else {r=inference->advance(xs,(index+1)*data.span);value=loss(r);}
    auto actual=detached(r);const auto number=value.cpu().item<double>();portable_torch::synchronize(device);return {std::move(actual),number};
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
    at::NoGradGuard ng;if(!training)throw std::invalid_argument("benchmark checkpoint needs training");
    save_resident(path,fixture(material,dtype),training->checkpoint(),limit);
  }
  void restore(const std::filesystem::path& path,int64_t limit)override{
    at::NoGradGuard ng;if(!training)throw std::invalid_argument("benchmark checkpoint needs training");
    training->close();training.reset();auto c=load_resident(path,data,limit);
    training=std::make_unique<WeightedTrainingSession>(data.graph,data.model,c,device,limits,node_owners);
  }
  void close()override{at::NoGradGuard ng;if(training)training->close();if(inference)inference->close();}
};
}
std::unique_ptr<Resident> resident(const Json& data,at::Device device,bool training,bool prefill,at::ScalarType dtype,
                                   const std::vector<at::Device>& peers){
  return std::make_unique<Candidate>(data,device,training,prefill,dtype,peers);
}
}
