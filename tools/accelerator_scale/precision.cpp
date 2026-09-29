#include "precision.h"
#include <stdexcept>
#include <set>

namespace accelerator_scale {
std::unique_ptr<NamedOptimizer> make_optimizer(ParameterRegistry& registry,const TrainingConfig& config) {
  OptimizerGroup group;group.lr=config.learning_rate;group.weight_decay=.01;
  group.eps=1e-5;group.momentum=.9;
  for(const auto& owner:registry.owners())group.parameters.push_back(owner.canonical);
  if(config.optimizer=="adamw")return std::make_unique<AdamW>(registry,std::vector<OptimizerGroup>{group});
  return std::make_unique<SGD>(registry,std::vector<OptimizerGroup>{group});
}

TrainingOwners::TrainingOwners(const pdg_scale::Fixture& f,const TrainingConfig& c)
    : payload_(f.owners),loss_scale_(c.loss_scale),half_(f.embedding.scalar_type()==at::kHalf) {
  if(c.backward_threads<1 || c.backward_threads>160 || c.optimizer_threads<1 || c.optimizer_threads>160)
    throw std::invalid_argument("CPU phase thread budget requires1..160");
  if(f.embedding.device().is_cpu()) {
    backward_threads_=c.backward_threads;optimizer_threads_=c.optimizer_threads;
  } else if(c.backward_threads!=1 || c.optimizer_threads!=1)
    throw std::invalid_argument("phase thread options require CPU training");
  at::NoGradGuard guard;
  std::set<const c10::TensorImpl*> identities;
  for(const auto& p:payload_)
    if(!identities.insert(p.unsafeGetTensorImpl()).second || p.scalar_type()!=f.embedding.scalar_type())
      throw std::invalid_argument("duplicate or mixed-dtype precision owner");
  for(size_t i=0;i<payload_.size();++i) {
    const auto& p=payload_[i];
    auto master=half_?p.detach().to(at::kFloat).set_requires_grad(true):p;
    masters_.push_back(master);registry_.add("owner."+std::to_string(i),master);
  }
  if(registry_.owners().size()!=payload_.size())throw std::runtime_error("duplicated precision owner");
  optimizer_=make_optimizer(registry_,c);
}
void TrainingOwners::zero_grad() {
  optimizer_->zero_grad(true);
  if(half_)for(auto& p:payload_)p.mutable_grad()=Tensor();
}
void TrainingOwners::backward(const Tensor& loss) {
  CpuPhaseThreads threads(backward_threads_);backward_counts_=threads.current();
  (loss*loss_scale_).backward();
}
void TrainingOwners::step() {
  CpuPhaseThreads threads(optimizer_threads_);optimizer_counts_=threads.current();
  at::NoGradGuard guard;
  // Preflight every gradient before updating any owner. One host extraction per
  // device; neither a nonfinite gradient nor an absent one is silently repaired.
  std::map<std::string,std::vector<Tensor>> checks;
  for(size_t i=0;i<payload_.size();++i) {
    auto g=payload_[i].grad();
    if(!g.defined()) { if(half_)masters_[i].mutable_grad()=Tensor();continue; }
    if(half_)masters_[i].mutable_grad()=g.to(at::kFloat)/loss_scale_;
    else if(loss_scale_!=1.)masters_[i].mutable_grad()=g/loss_scale_;
    checks[g.device().str()].push_back(at::isfinite(masters_[i].grad()).all());
  }
  for(const auto& [device,values]:checks)
    if(!at::stack(values).all().item<bool>())
      throw std::runtime_error("nonfinite training gradient; reduce --loss-scale or use float32");
  optimizer_->step();
  checks.clear();
  for(size_t i=0;i<payload_.size();++i)if(masters_[i].grad().defined()) {
    if(half_)payload_[i].copy_(masters_[i]);
    checks[payload_[i].device().str()].push_back(at::isfinite(payload_[i]).all());
  }
  for(const auto& [device,values]:checks)
    if(!at::stack(values).all().item<bool>())
      throw std::runtime_error("updated payload overflow; restore checkpoint and change precision/lr");
}
double TrainingOwners::master_bytes() const {
  double result=0;if(half_)for(const auto& p:masters_)result+=p.numel()*p.element_size();return result;
}
}  // namespace accelerator_scale
