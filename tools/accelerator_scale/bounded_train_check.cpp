#include "bounded_check.h"
#include "bounded_optimizer.h"
#include "graph_replay.h"
#include "precision.h"
#include "../../cpp/bench/streaming.h"
#include <iostream>
#include <limits>

namespace accelerator_scale::bounded {
namespace {
struct Step {
  Window window;
  Value loss;
  std::vector<Tensor> gradients,payload;
  std::vector<OptimizerState> optimizer;
  Tensor finite;
};
void check_optimizer_guards(at::Device device,const std::string& kind) {
  auto owner=at::ones({2},at::TensorOptions().dtype(at::kFloat).device(device)).set_requires_grad(true);
  Optimizer optimizer({owner},kind,1e-4,1.,1);
  auto used=at::zeros({1,1},owner.options().dtype(at::kBool));
  Value objective{owner.sum(),used};
  auto assert_state=[&](Index step,const Tensor& expected) {
    auto state=optimizer.snapshot();
    if(state[0].step.item<Index>()!=step || !at::equal(owner,expected))
      throw std::runtime_error("bounded optimizer rejected update changed owner/counter");
  };
  auto initial=owner.detach().clone();
  if(!optimizer.step(objective,{Tensor()}).item<bool>())throw std::runtime_error("absent owner rejected");
  assert_state(0,initial);
  used.fill_(true);
  if(optimizer.step(objective,{Tensor()}).item<bool>())throw std::runtime_error("connected missing gradient accepted");
  assert_state(0,initial);
  if(!optimizer.step(objective,{at::zeros_like(owner)}).item<bool>())throw std::runtime_error("connected zero rejected");
  if(optimizer.snapshot()[0].step.item<Index>()!=1 || at::equal(owner,initial))
    throw std::runtime_error("connected zero failed to apply weight decay/counter");
  auto updated=owner.detach().clone();
  if(optimizer.step(objective,{at::zeros_like(owner)}).item<bool>())throw std::runtime_error("optimizer capacity overflow accepted");
  assert_state(1,updated);
  optimizer.reset();
  if(optimizer.step(objective,{at::full_like(owner,std::numeric_limits<float>::infinity())}).item<bool>())
    throw std::runtime_error("nonfinite gradient accepted");
  assert_state(0,initial);
  std::cout<<"PASS optimizer absence/zero/missing/nonfinite/capacity guards kind="<<kind<<'\n'<<std::flush;
}
}
void check_training(pdg_scale::Config c,const pdg_scale::Topology& topology,at::Device device,Index devices,bool capture) {
  if(c.width>64 || c.batch>8 || c.steps>6)throw std::invalid_argument("bounded training parity requires small tensors");
  at::AutoGradMode enabled(true);const bool half=c.runtime.dtype==at::kHalf;
  const double rtol=c.check_rtol,atol=c.check_atol;
  auto initial=(at::arange(c.steps).unsqueeze(1)*7+at::arange(c.batch).unsqueeze(0)*3).remainder(c.vocab).to(at::kLong);
  for(const std::string kind:{"sgd","adamw"}) {
    check_optimizer_guards(device,kind);
    std::unique_ptr<GraphReplay> replay;
    if(capture){std::vector<at::Device> targets;for(Index i=0;i<devices;++i)targets.emplace_back(device.type(),device.index()+i);
      replay=std::make_unique<GraphReplay>(targets);}
    auto actual=fixture(c,topology,true);auto placement=place(actual,device,devices,"locality",true);
    placement.scoring={"model","model",at::kFloat};
    Program p(actual,topology,placement,Limits{c.steps,c.batch,int64_t(4)<<30,true,true});
    Optimizer optimizer(actual.owners,kind,1e-4,half?128.:1.);
    auto ids=initial.to(device);Step step;
    auto execute=[&] {
      step.window=p.run(ids);step.loss=p.loss(step.window,ids);
      step.gradients=p.vjp(step.loss,at::ones_like(step.loss.data)*(half?128.:1.),false);
      step.finite=optimizer.step(step.loss,step.gradients);step.optimizer=optimizer.snapshot();
      step.payload.clear();
      {at::NoGradGuard guard;for(const auto& x:actual.owners)step.payload.push_back(x.clone());}
    };
    if(capture) {execute();synchronize(placement);optimizer.reset();replay->capture(execute);}
    for(Index trial=0;trial<(capture?3:1);++trial) {
      optimizer.reset();auto input=(initial+trial*5).remainder(c.vocab);ids.copy_(input);synchronize(placement);
      auto ref=fixture(c,topology,false);
      TrainingConfig config;config.optimizer=kind;config.loss_scale=half?128.:1.;
      TrainingOwners reference_owners(ref,config);
      for(Index i=0;i<3;++i) {
        // The benchmark replays one update with persistent owners. Exercise
        // changing device optimizer counters through that exact replay shape.
        if(capture){replay->replay();replay->synchronize();}else {execute();synchronize(placement);}
        if(!step.finite.item<bool>())throw std::runtime_error("bounded nonfinite training gradient/payload");
        reference_owners.zero_grad();auto expected=oracle(ref,c,topology,input);
        std::vector<Tensor> losses;
        for(Index t=0;t<c.steps;++t)losses.push_back(at::cross_entropy_loss(expected.logits[t].to(at::kFloat),(input[t]+1).remainder(c.vocab)));
        auto loss=at::stack(losses).mean();reference_owners.backward(loss);
        try {tide_bench::compare(export_result(p,step.window),expected.result,true,c.runtime.dtype,std::nullopt,rtol,atol);}
        catch(const std::exception& e){throw std::runtime_error(kind+" update "+std::to_string(i)+" trial "+std::to_string(trial)+": "+e.what());}
        close(step.loss.data,loss,rtol,atol,"training loss");
        auto gradients=export_gradients(step.loss,step.gradients);
        for(size_t j=0;j<gradients.size();++j)
          close(gradients[j].defined()?gradients[j].to(at::kFloat)/config.loss_scale:Tensor(),
            ref.owners[j].grad().defined()?ref.owners[j].grad().to(at::kFloat)/config.loss_scale:Tensor(),rtol,atol,"training gradient "+std::to_string(j));
        reference_owners.step();
        for(size_t j=0;j<actual.owners.size();++j) {
          const auto& s=step.optimizer[j];
          close(step.payload[j],ref.owners[j],rtol,atol,"payload owner "+std::to_string(j));
          close(s.master,reference_owners.masters()[j],rtol,atol,"master owner "+std::to_string(j));
          auto it=reference_owners.optimizer().state().find("owner."+std::to_string(j));
          const auto used=s.step.item<Index>();
          if((used>0)!=(it!=reference_owners.optimizer().state().end()))throw std::runtime_error("optimizer None/zero slot inventory");
          if(!used)continue;
          if(kind=="sgd")close(s.first,it->second.momentum_buffer,rtol,atol,"momentum");
          else {
            if(used!=it->second.step)throw std::runtime_error("optimizer exact step count");
            close(s.first,it->second.exp_avg,rtol,atol,"first moment");close(s.second,it->second.exp_avg_sq,rtol,atol,"second moment");
          }
        }
      }
      std::cout<<"PASS bounded three complete updates optimizer="<<kind<<" replay="<<capture<<" trial="<<trial<<" memory="<<c.memory<<'\n'<<std::flush;
    }
  }
}
}  // namespace accelerator_scale::bounded
