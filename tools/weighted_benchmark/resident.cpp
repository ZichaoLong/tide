#include "benchmark.h"
#include "tide/resident_weighted_training.h"
#include "tide/parameters.h"
#include "portable_torch/runtime.hpp"
#include <torch/csrc/autograd/autograd.h>
namespace weighted_benchmark {
using namespace tide::device_online;
namespace {
class Candidate final:public Resident {
  Fixture data;at::Device device;std::unique_ptr<WeightedFlow> inference;std::unique_ptr<WeightedTrainingSession> training;
 public:
  Candidate(const Json& j,at::Device d,bool train,bool prefill):data(fixture(j)),device(d){
    at::NoGradGuard ng;Continuation q;q.identity=data.graph.identity;q.batch_size=data.samples;
    WeightedTrainingLimits l;const auto& x=j.at("limits");
#define FIELD(name) l.forward.name=x.at(#name).get<decltype(l.forward.name)>()
    FIELD(queue);FIELD(stages);FIELD(cache);FIELD(tensor_bytes);FIELD(operator_bytes);FIELD(timeout_ms);
#undef FIELD
#define FIELD(name) l.name=x.at(#name).get<decltype(l.name)>()
    FIELD(retained_bytes);FIELD(backward_bytes);FIELD(optimizer_bytes);FIELD(total_bytes);
#undef FIELD
    l.forward.prefill=prefill;
    if(train){OptimizerGroup group;for(const auto& p:data.model.parameters().owners())group.parameters.push_back(p.canonical);
      group.lr=.001;group.momentum=.3;group.weight_decay=.01;
      training=std::make_unique<WeightedTrainingSession>(data.graph,data.model,q,d,ResidentOptimizerKind::sgd,std::vector<OptimizerGroup>{group},l);
    }else inference=std::make_unique<WeightedFlow>(data.graph,data.model,q,d,l.forward);
  }
  Step step(Index index)override{
    at::NoGradGuard ng;auto xs=inputs(data,index);Result r;Tensor value;
    if(training){auto w=training->advance(xs,(index+1)*data.span);const auto& f=w.fields.at("outputs");Tensor bar;
      {at::AutoGradMode grad(true);auto x=f.values.detach().to(at::kFloat).clone().set_requires_grad(true);
        auto numeric=x.narrow(-1,0,1)*x.narrow(-1,1,x.size(-1)-1);
        value=at::where(f.present.unsqueeze(-1),numeric.square(),at::zeros_like(numeric)).sum()*.001;
        bar=torch::autograd::grad({value},{x})[0];}
      WeightedCotangents roots;roots.token=w.token;roots.fields["outputs"]={bar,f.present};r=training->result();
      auto gradients=training->backward({roots});if(!at::isfinite(gradients.parameters.values).all().item<bool>())throw std::runtime_error("nonfinite resident gradient");
      if(!training->step().applied)throw std::runtime_error("resident benchmark optimizer refused");
    }else {r=inference->advance(xs,(index+1)*data.span);value=loss(r);}
    auto actual=detached(r);const auto number=value.cpu().item<double>();portable_torch::synchronize(device);return {std::move(actual),number};
  }
  std::map<std::string,Tensor> parameters()const override{
    at::NoGradGuard ng;return training?training->checkpoint().parameters:std::map<std::string,Tensor>{};
  }
  void close()override{at::NoGradGuard ng;if(training)training->close();if(inference)inference->close();}
};
}
std::unique_ptr<Resident> resident(const Json& data,at::Device device,bool training,bool prefill){
  return std::make_unique<Candidate>(data,device,training,prefill);
}
}
