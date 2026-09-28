#include "training.h"
#include "../../cpp/bench/streaming.h"
#include <iostream>

namespace accelerator_scale {
namespace {
void close(const Tensor& a,const Tensor& b,const std::string& context) {
  if(a.defined()!=b.defined())throw std::runtime_error("training None/zero mismatch: "+context);
  if(!a.defined())return;
  auto x=a.detach().to(at::kCPU),y=b.detach().to(at::kCPU);
  if(x.sizes()!=y.sizes() || x.scalar_type()!=y.scalar_type()
      || !at::isfinite(x).all().item<bool>() || !at::isfinite(y).all().item<bool>()
      || !at::allclose(x,y,1e-5,1e-6))
    throw std::runtime_error("training value mismatch: "+context);
}
struct Side {
  pdg_scale::Fixture fixture;
  Placement placement;
  Options options;
  ParameterRegistry registry;
  std::unique_ptr<NamedOptimizer> optimizer;
  Side(pdg_scale::Config c,const pdg_scale::Topology& topology,at::Device device,
       Index devices,bool candidate,const Placement& policies,const TrainingConfig& training) {
    portable_torch::seed_runtime(at::Device(at::kCPU),c.runtime.seed);
    c.emission=candidate?"row":"slot";fixture=pdg_scale::fixture(c,topology);
    configure_scoring(fixture,policies.scoring);
    if(!candidate) {
      std::map<const c10::StorageImpl*,Tensor> owners;
      for(const auto& p:fixture.owners)owners[p.storage().unsafeGetStorageImpl()]=p;
      for(auto& w:fixture.model.nodes)for(auto& [name,p]:w.extra) {
        const auto it=owners.find(p.storage().unsafeGetStorageImpl());
        if(it!=owners.end() && !p.is_same(it->second))p=it->second.as_strided(p.sizes(),p.strides(),p.storage_offset());
      }
    }
    if(candidate)placement=place(fixture,device,devices,policies.policy,policies.resident);
    placement.scoring=policies.scoring;
    if(candidate){placement.ranking_device=policies.ranking_device;placement.event_device=policies.event_device;}
    options.trace=true;options.packed=candidate;options.workers=candidate?c.workers:1;
    options.parallel_regions=candidate&&c.parallel_regions;options.packed_sources=candidate&&c.packed_sources;
    options.batch_next=candidate&&c.batch_next;
    options.full_autograd=candidate?c.full_autograd:"replay";
    options.aggregate_autograd=candidate?c.aggregate_autograd:"replay";
    register_owners(registry,fixture);optimizer=make_optimizer(registry,training);
  }
};
}
void check_training(const pdg_scale::Config& c,const pdg_scale::Topology& topology,at::Device device,
                    Index count,const Placement& policies,const TrainingConfig& training) {
  if(c.width>64 || c.batch>8 || c.steps>6)throw std::invalid_argument("training parity requires bounded tensors");
  at::AutoGradMode grad(true);
  Side expected(c,topology,at::Device(at::kCPU),1,false,policies,training);
  Side actual(c,topology,device,count,true,policies,training);
  for(Index step=0;step<3;++step) {
    expected.optimizer->zero_grad();actual.optimizer->zero_grad();
    auto a=training_window(c,topology,expected.fixture,expected.placement,expected.options);
    auto b=training_window(c,topology,actual.fixture,actual.placement,actual.options);
    synchronize(actual.placement);tide_bench::compare(b.result,a.result,true,at::kFloat);
    close(b.loss,a.loss,"loss");a.loss.backward();b.loss.backward();synchronize(actual.placement);
    for(size_t i=0;i<actual.fixture.owners.size();++i)
      close(actual.fixture.owners[i].grad(),expected.fixture.owners[i].grad(),"gradient owner "+std::to_string(i));
    expected.optimizer->step();actual.optimizer->step();synchronize(actual.placement);
    for(size_t i=0;i<actual.fixture.owners.size();++i)
      close(actual.fixture.owners[i],expected.fixture.owners[i],"updated owner "+std::to_string(i));
    const auto& x=expected.optimizer->state();const auto& y=actual.optimizer->state();
    if(x.size()!=y.size())throw std::runtime_error("training optimizer state inventory");
    for(const auto& [name,u]:x) {
      const auto& v=y.at(name);if(u.step!=v.step)throw std::runtime_error("optimizer step mismatch");
      close(v.momentum_buffer,u.momentum_buffer,name+" momentum");
      close(v.exp_avg,u.exp_avg,name+" first moment");close(v.exp_avg_sq,u.exp_avg_sq,name+" second moment");
    }
  }
  std::cout<<"CHECK training three complete windows, independent scalar schedule, loss/gradients/None/updates/optimizer slots: passed\n"<<std::flush;
}
}  // namespace accelerator_scale
