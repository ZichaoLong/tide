#include "flow_config.h"
#include "bounded_optimizer.h"
#include "graph_replay.h"
#include "precision.h"
#include "flow_roots.h"
#include "peer_transport.h"
#include "../../cpp/bench/streaming.h"
#include <torch/csrc/autograd/autograd.h>
#include <iostream>

namespace accelerator_scale::flows {
void check_inactive_full(const Config&,const Topology&,at::Device);
namespace {
void close(const Tensor& a,const Tensor& b,double rtol,double atol,const std::string& name) {
  if(a.defined()!=b.defined())throw std::runtime_error("flow None/zero mismatch: "+name);
  if(!a.defined())return;
  auto x=a.detach().to(at::kCPU).to(at::kDouble),y=b.detach().to(at::kCPU).to(at::kDouble);
  if(x.sizes()!=y.sizes() || !at::isfinite(x).all().item<bool>() || !at::allclose(x,y,rtol,atol))
    throw std::runtime_error("flow value mismatch: "+name+" max_error="+std::to_string((x-y).abs().max().item<double>()));
}
Options oracle_options(const Config& c) {
  auto o=options(c,true);o.packed=false;o.workers=1;o.full_autograd="replay";o.aggregate_autograd="replay";
  o.packed_sources=false;o.batch_next=false;o.parallel_regions=false;o.compact_events=false;o.defer_state_release=false;
  return o;
}
}  // namespace
Fixture seeded(const Config& c,const Topology& topology,bool rows) {
  portable_torch::seed_runtime(at::Device(at::kCPU),c.model.runtime.seed);
  auto f=fixture(c.model,topology,rows);configure_scoring(f.values,c.scoring);
  // The direct Settle body owns the same tensors but also needs the declared
  // custom Read implementation before the native encoder compiles its identity.
  pdg_scale::Fixture body;body.graph=f.body;body.model=f.body_model;
  configure_scoring(body,c.scoring);f.body=body.graph;f.body_model=body.model;
  if(f.settle)f.settle=std::make_shared<SettleGraph>(f.body,topology.ranks);
  f.values.graph.compile();tide::configure_model(f.values.graph,f.values.model);return f;
}
Placement placed(Fixture& f,const Config& c,at::Device device) {
  auto p=place(f.values,device,c.devices,c.partition,true);p.scoring=c.scoring;
  p.ranking_device=c.ranking;p.event_device=c.events;return p;
}
void check(const Config& c,const Topology& topology,at::Device device) {
  const bool half=c.model.runtime.dtype==at::kHalf,bounded=c.scheduler.rfind("bounded-",0)==0;
  const double rtol=half?.02:c.model.runtime.dtype==at::kDouble?1e-8:1e-5;
  const double atol=half?.004:c.model.runtime.dtype==at::kDouble?1e-10:1e-6;
  Config cfg=c;cfg.model.grad=true;
  auto initial=(at::arange(c.model.steps).unsqueeze(1)*7+at::arange(c.model.batch).unsqueeze(0)*3).remainder(c.model.vocab).to(at::kLong);
  // Independent scalar CPU Streaming oracle and per-slot projection formula.
  // The candidate never receives expected routes or values.
  {
    at::AutoGradMode grad(true);auto reference=seeded(cfg,topology,false),candidate=seeded(cfg,topology,true);
    auto reference_config=cfg;reference_config.devices=1;reference_config.model.packed=false;
    auto cpu=placed(reference,reference_config,at::Device(at::kCPU));auto placement=placed(candidate,cfg,device);
    auto ref_options=oracle_options(reference_config);
    std::vector<Tensor> ri,ci;
    for(Index t=0;t<c.model.steps;++t) {
      ri.push_back(at::zeros({c.model.batch,c.model.width},reference.values.embedding.options()).set_requires_grad(true));
      ci.push_back(at::zeros({c.model.batch,c.model.width},candidate.values.embedding.options()).set_requires_grad(true));
    }
    auto expected=host_window(cfg.model,reference,cpu,"streaming",ref_options,initial,ri);
    auto leaves=reference.values.owners;leaves.insert(leaves.end(),ri.begin(),ri.end());
    auto candidate_leaves=candidate.values.owners;candidate_leaves.insert(candidate_leaves.end(),ci.begin(),ci.end());
    std::unique_ptr<bounded::Program> program;bounded::Window window;HostWindow actual;Result observed;
    std::unique_ptr<PeerTransport> peer;
    if(bounded && device.type()==c10::DeviceType::PrivateUse1 && c.devices>1)peer=std::make_unique<PeerTransport>(placement.devices);
    const bounded::Limits limits{c.model.steps,c.model.batch,int64_t(16)<<30,true,true};
    if(bounded) {
      program=std::make_unique<bounded::Program>(candidate.values,schedule(candidate,limits,true),placement,limits);
      window=program->run(initial.to(candidate.values.embedding.device()),ci);synchronize(placement);
      observed=bounded::export_result(*program,window);
      for(const auto& x:window.logits)actual.logits.push_back(x.data);
    } else {
      actual=host_window(cfg.model,candidate,placement,c.scheduler,options(cfg,true),initial,ci);synchronize(placement);
      observed=actual.result;
    }
    tide_bench::compare(observed,expected.result,true,c.model.runtime.dtype,std::nullopt,rtol,atol);
    check_roots(expected.result,observed,program.get(),window,leaves,candidate_leaves,ci,rtol,atol);
    for(Index t=0;t<c.model.steps;++t) {
      close(actual.logits[t],expected.logits[t],rtol,atol,"logits");
      for(Index direction=0;direction<3;++direction) {
        auto index=at::arange(expected.logits[t].numel(),at::TensorOptions().dtype(at::kLong)).reshape(expected.logits[t].sizes());
        auto u=((index*(direction+1)+t).remainder(7+4*direction)-(3+2*direction)).to(c.model.runtime.dtype)/8.;
        if(direction==2)u=at::zeros_like(u);
        auto bg=torch::autograd::grad({expected.logits[t]},leaves,{u},true,false,true);
        std::vector<Tensor> ag;
        if(bounded) {
          auto root=bounded::root(window.logits[t].data,window.logits[t].dependencies);
          ag=bounded::export_gradients(root,program->vjp(root,u.to(root.data.device()),true,ci));
        } else ag=torch::autograd::grad({actual.logits[t]},candidate_leaves,{u.to(actual.logits[t].device())},true,false,true);
        for(size_t j=0;j<bg.size();++j)close(ag[j],bg[j],rtol,atol,"isolated VJP "+std::to_string(j));
      }
    }
    std::cout<<"PASS complete-flow observables and independent VJPs family="<<c.family<<" scheduler="<<c.scheduler<<'\n'<<std::flush;
  }
  const bool capture=c.scheduler=="bounded-replay";
  if(bounded)check_inactive_full(c,topology,device);
  if(!c.training && !capture)return;
  at::AutoGradMode grad(c.training);
  std::unique_ptr<GraphReplay> replay;
  if(capture){std::vector<at::Device> targets;for(Index i=0;i<c.devices;++i)targets.emplace_back(device.type(),device.index()+i);
    replay=std::make_unique<GraphReplay>(targets);}
  auto candidate=seeded(c,topology,true);auto placement=placed(candidate,c,device);
  std::unique_ptr<PeerTransport> eager_peer;
  if(bounded && !capture && device.type()==c10::DeviceType::PrivateUse1 && c.devices>1)
    eager_peer=std::make_unique<PeerTransport>(placement.devices);
  const bounded::Limits limits{c.model.steps,c.model.batch,int64_t(16)<<30,true,c.training};
  std::unique_ptr<bounded::Program> program;
  if(bounded)program=std::make_unique<bounded::Program>(candidate.values,schedule(candidate,limits,c.training),placement,limits);
  auto ids=initial.to(candidate.values.embedding.device());bounded::Window window;bounded::Value loss;HostWindow actual;
  std::vector<Tensor> gradients;Tensor finite;
  TrainingConfig train;train.optimizer=c.optimizer;train.loss_scale=half?128.:1.;
  std::unique_ptr<bounded::Optimizer> optimizer;std::unique_ptr<TrainingOwners> native;
  if(c.training) {
    if(bounded)optimizer=std::make_unique<bounded::Optimizer>(candidate.values.owners,c.optimizer,1e-4,train.loss_scale);
    else native=std::make_unique<TrainingOwners>(candidate.values,train);
  }
  auto execute=[&] {
    if(bounded) {
      window=program->run(ids);
      if(c.training){loss=program->loss(window,ids);gradients=program->vjp(loss,at::ones_like(loss.data)*train.loss_scale,false);
        finite=optimizer->step(loss,gradients);}
    } else {
      if(native)native->zero_grad();actual=host_window(c.model,candidate,placement,c.scheduler,options(c,true),ids);
      if(native){native->backward(actual.loss);native->step();}
    }
  };
  if(capture){execute();synchronize(placement);if(optimizer)optimizer->reset();replay->capture(execute);}
  const auto trials=capture?3:1;
  for(Index trial=0;trial<trials;++trial) {
    if(optimizer)optimizer->reset();auto input=(initial+trial*5).remainder(c.model.vocab);ids.copy_(input);synchronize(placement);
    auto reference=seeded(c,topology,false);auto rc=c;rc.devices=1;rc.model.packed=false;
    auto cpu=placed(reference,rc,at::Device(at::kCPU));std::unique_ptr<TrainingOwners> ref_owners;
    if(c.training)ref_owners=std::make_unique<TrainingOwners>(reference.values,train);
    auto ro=oracle_options(rc);
    for(Index step=0;step<(c.training?3:1);++step) {
      if(capture){replay->replay();replay->synchronize();}else{execute();synchronize(placement);}
      if(optimizer && !finite.item<bool>())throw std::runtime_error("nonfinite bounded flow update");
      if(ref_owners)ref_owners->zero_grad();auto expected=host_window(c.model,reference,cpu,"streaming",ro,input);
      auto result=bounded?bounded::export_result(*program,window):actual.result;
      tide_bench::compare(result,expected.result,true,c.model.runtime.dtype,std::nullopt,rtol,atol);
      for(Index t=0;t<c.model.steps;++t)close(bounded?window.logits[t].data:actual.logits[t],expected.logits[t],rtol,atol,"replayed logits");
      if(c.training) {
        close(bounded?loss.data:actual.loss,expected.loss,rtol,atol,"loss");ref_owners->backward(expected.loss);
        auto gg=bounded?bounded::export_gradients(loss,gradients):std::vector<Tensor>{};
        auto unscaled=[&](const Tensor& g) {
          if(!g.defined())return Tensor();
          return (half?g.to(at::kFloat):g)/train.loss_scale;
        };
        for(size_t i=0;i<reference.values.owners.size();++i)
          close(unscaled(bounded?gg[i]:candidate.values.owners[i].grad()),unscaled(reference.values.owners[i].grad()),rtol,atol,"training gradient "+std::to_string(i));
        ref_owners->step();
        for(size_t i=0;i<reference.values.owners.size();++i)close(candidate.values.owners[i],reference.values.owners[i],rtol,atol,"updated owner "+std::to_string(i));
        if(optimizer) {
          auto states=optimizer->snapshot();
          for(size_t i=0;i<states.size();++i) {
            auto it=ref_owners->optimizer().state().find("owner."+std::to_string(i));const auto used=states[i].step.item<Index>();
            if((used>0)!=(it!=ref_owners->optimizer().state().end()))throw std::runtime_error("optimizer structural slot inventory");
            close(states[i].master,ref_owners->masters()[i],rtol,atol,"master");
            if(!used)continue;
            if(c.optimizer=="sgd")close(states[i].first,it->second.momentum_buffer,rtol,atol,"momentum");
            else {if(used!=it->second.step)throw std::runtime_error("optimizer int64 count mismatch");
              close(states[i].first,it->second.exp_avg,rtol,atol,"first moment");close(states[i].second,it->second.exp_avg_sq,rtol,atol,"second moment");}
          }
        } else {
          for(size_t i=0;i<native->masters().size();++i)close(native->masters()[i],ref_owners->masters()[i],rtol,atol,"native master");
          const auto& x=native->optimizer().state();const auto& y=ref_owners->optimizer().state();
          if(x.size()!=y.size())throw std::runtime_error("native optimizer slot inventory");
          for(const auto& [name,s]:x) {
            const auto& r=y.at(name);if(s.step!=r.step)throw std::runtime_error("native optimizer int64 step mismatch");
            close(s.momentum_buffer,r.momentum_buffer,rtol,atol,"native momentum");
            close(s.exp_avg,r.exp_avg,rtol,atol,"native first moment");
            close(s.exp_avg_sq,r.exp_avg_sq,rtol,atol,"native second moment");
          }
        }
      }
    }
    std::cout<<"PASS complete-flow replay/updates family="<<c.family<<" training="<<c.training<<" trial="<<trial<<'\n'<<std::flush;
  }
}
}  // namespace accelerator_scale::flows
