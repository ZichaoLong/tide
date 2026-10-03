#pragma once
#include "consumer.h"
#include <ATen/core/grad_mode.h>
#include <torch/csrc/autograd/autograd.h>
#include <cmath>

namespace tide_flow {
// Consumer policy only. The public NamedOptimizer/checkpoint interface keeps
// its own dtype contract. Each canonical half leaf has exactly one FP32 master.
class EagerOptimizer {
 public:
  EagerOptimizer(tide::ParameterRegistry& payload,const std::string& kind,
                 bool half,double scale,tide::OptimizerGroup group)
      : payload_(payload.owners()),half_(half),scale_(scale) {
    if(!std::isfinite(scale)||scale<=0||(!half&&scale!=1))
      throw std::invalid_argument("loss-scale requires positive finite FP16 training scale");
    for(const auto& owner:payload_) {
      if(half) {
        if(owner.value.scalar_type()!=at::kHalf||!owner.value.is_leaf()||!owner.value.requires_grad())
          throw std::invalid_argument("master optimizer requires trainable FP16 leaves");
        auto master=owner.value.detach().to(at::kFloat).set_requires_grad(true);
        for(const auto& alias:owner.aliases)masters_.add(alias,master);
      }
    }
    auto& registry=half?masters_:payload;
    if(group.parameters.empty())for(const auto& owner:payload_)group.parameters.push_back(owner.canonical);
    if(kind=="sgd")inner_=std::make_unique<tide::SGD>(registry,std::vector<tide::OptimizerGroup>{group});
    else if(kind=="adamw")inner_=std::make_unique<tide::AdamW>(registry,std::vector<tide::OptimizerGroup>{group});
    else throw std::invalid_argument("unknown eager optimizer");
  }
  void zero_grad() {
    inner_->zero_grad();
    if(half_)for(auto& owner:payload_)owner.value.mutable_grad().reset();
  }
  void backward(const Tensor& loss) {
    if(half_&&(loss.scalar_type()!=at::kFloat||loss.numel()!=1))
      throw std::invalid_argument("FP16 training requires a scalar FP32 loss");
    if(half_)(loss*scale_).backward();else loss.backward();
  }
  void step() {
    at::NoGradGuard guard;
    if(half_) {
      std::map<std::string,std::vector<Tensor>> checks;
      for(const auto& owner:payload_) {
        auto master=masters_.value(owner.canonical);const auto grad=owner.value.grad();
        master.mutable_grad()=grad.defined()?grad.to(at::kFloat)/scale_:Tensor();
        if(master.grad().defined())checks[master.device().str()].push_back(at::isfinite(master.grad()).all());
      }
      finite(checks,"nonfinite gradient; optimizer not applied");
    }
    inner_->step();
    if(half_) {
      std::map<std::string,std::vector<Tensor>> checks;
      for(const auto& owner:payload_)if(owner.value.grad().defined()) {
        owner.value.copy_(masters_.value(owner.canonical));
        checks[owner.value.device().str()].push_back(at::isfinite(owner.value).all());
      }
      // Match the public Python master policy: cast overflow explicitly fails;
      // callers must restore/restart, never silently retry or skip an update.
      finite(checks,"updated FP16 payload overflow; restart with changed precision/lr");
    }
  }
  const tide::NamedOptimizer& inner() const {return *inner_;}
 private:
  static void finite(const std::map<std::string,std::vector<Tensor>>& checks,const char* error) {
    for(const auto& [_,values]:checks)if(!at::stack(values).all().item<bool>())throw std::runtime_error(error);
  }
  std::vector<tide::ParameterOwner> payload_;
  bool half_;double scale_;
  tide::ParameterRegistry masters_;
  std::unique_ptr<tide::NamedOptimizer> inner_;
};
} // namespace tide_flow
