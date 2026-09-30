#include "training_test.h"
#include <ATen/core/grad_mode.h>
#include <algorithm>
#include <iomanip>
#include <iostream>

namespace tide::device_online::test {
void train_numerics(at::Device device,Fixture f,bool prefill) {
  at::NoGradGuard guard;f.model=train_model(f.model,at::kFloat);
  OptimizerGroup group;group.lr=.001;group.weight_decay=.0125;group.amsgrad=true;group.eps=1e-8;
  for(const auto& owner:f.model.parameters(true).owners())group.parameters.push_back(owner.canonical);
  ResidentTrainingLimits limits;limits.forward.prefill=prefill;limits.forward.trace=512;
  limits.forward.full_chunk_rows=3;limits.reverse_chunk_rows=3;
  ResidentTrainingSession candidate(f.graph,f.model,f.initial,device,ResidentOptimizerKind::adamw,{group},limits);
  std::vector<ResidentCotangents> roots;Index cut=f.initial.cut;int index=0;
  for(auto stop:retained_stops(cut)) {
    std::vector<External> input;for(const auto& x:f.input)if(cut<=x.time&&x.time<stop)input.push_back(x);
    roots.push_back(train_roots(candidate.advance(input,stop,stop),index++,4));cut=stop;
  }
  const auto gradient=candidate.backward(roots);
  // This is an observation of the candidate, used only by the CPU oracle.
  // No CPU-computed gradient, value or route is fed back into candidate.
  auto values=gradient.values.cpu(),connected=gradient.connected.cpu();
  train_require(candidate.step().applied,"near-zero device optimizer refused finite input");
  const auto checkpoint=candidate.checkpoint();
  std::map<std::string,Tensor> prior;
  for(auto dtype:{at::kFloat,at::kDouble}) {
    auto independent=f;independent.model=train_model(f.model,dtype);independent.initial=train_boundary(f.initial,dtype);
    const auto reference=retained_reference(independent,4,dtype);
    train_gradients(gradient,reference,independent);
    auto same_gradient_model=train_model(f.model,dtype);
    auto same_registry=same_gradient_model.parameters(true),independent_registry=independent.model.parameters(true);
    AdamW same_optimizer(same_registry,{group}),independent_optimizer(independent_registry,{group});
    for(size_t i=0;i<gradient.names.size();++i) {
      const auto& name=gradient.names[i];auto owner=same_registry.value(name);
      if(connected[i].item<bool>())owner.mutable_grad()=values.narrow(0,gradient.offsets[i],owner.numel()).reshape(owner.sizes()).to(dtype);
      independent_registry.value(name).mutable_grad()=reference.gradients.at(name);
    }
    same_optimizer.step();independent_optimizer.step();
    train_checkpoint(checkpoint,same_gradient_model,same_optimizer);
    double maximum=0;bool within=true;
    for(size_t i=0;i<gradient.names.size();++i) {
      const auto& name=gradient.names[i];const auto actual=checkpoint.parameters.at(name),expected=independent_registry.value(name).to(at::kFloat);
      const auto difference=(actual-expected).abs().max().item<double>();maximum=std::max(maximum,difference);
      const bool close=at::allclose(actual,expected,1e-5,1e-6);within=within&&close;
      if(!close&&actual.numel()==1)std::cout<<std::setprecision(17)
        <<"training-conditioning owner="<<name<<" prefill="<<prefill<<" reference="<<dtype
        <<" device_gradient="<<values[gradient.offsets[i]].item<double>()
        <<" reference_gradient="<<reference.gradients.at(name).item<double>()
        <<" device_parameter="<<actual.item<double>()<<" reference_parameter="<<independent_registry.value(name).item<double>()
        <<" other_cpu_parameter="<<(prior.count(name)?prior.at(name).item<double>():expected.item<double>())<<'\n';
      prior[name]=expected.clone();
    }
    std::cout<<"training-conditioning: passed scope=VJP_and_same_gradient_optimizer_only prefill="<<prefill
      <<" reference="<<dtype<<" adam_epsilon="<<group.eps<<" independent_trajectory_within_tolerance="<<within
      <<" independent_trajectory_max_abs="<<maximum<<'\n';
  }
  candidate.close();
}
} // namespace tide::device_online::test
