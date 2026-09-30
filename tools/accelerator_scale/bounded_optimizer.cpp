#include "bounded_optimizer.h"
#include <cmath>
#include <stdexcept>

namespace accelerator_scale::bounded {
Optimizer::Optimizer(const std::vector<Tensor>& payload,std::string kind,double lr,double scale,Index max_updates)
    : payload_(payload),kind_(kind),lr_(lr),scale_(scale),max_updates_(max_updates) {
  if((kind!="sgd" && kind!="adamw") || !std::isfinite(lr) || lr<=0 || !std::isfinite(scale) || scale<=0
      || max_updates<1 || max_updates>100000)
    throw std::invalid_argument("invalid bounded optimizer configuration");
  at::NoGradGuard guard;
  for(const auto& p:payload) {
    if(p.scalar_type()!=at::kFloat && p.scalar_type()!=at::kHalf)
      throw std::invalid_argument("bounded optimizer requires FP32/FP16 payload, FP32 masters");
    // Reset is outside capture. Keep its immutable checkpoint on the host;
    // FP32 payload already provides the master storage used after backward.
    initial_.push_back(p.detach().to(at::kCPU).clone());
    auto master=p.scalar_type()==at::kFloat?p.detach():p.detach().to(at::kFloat);
    states_.push_back({master,at::zeros_like(master),at::zeros_like(master),at::zeros({},p.options().dtype(at::kLong))});
    if(!corrections_.count(p.device().str())) {
      // The reference evaluates bias correction in host double then casts its
      // scalars to FP32. Build that finite table once; device counters select
      // entries without scalar extraction or loss of int64 owner history.
      std::vector<float> first(max_updates+1),second(max_updates+1);
      for(Index i=0;i<=max_updates;++i){const auto t=std::max(Index(1),i);
        first[i]=lr/(1-std::pow(.9,double(t)));second[i]=std::sqrt(1-std::pow(.999,double(t)));}
      corrections_[p.device().str()]={at::tensor(first).to(p.device()),at::tensor(second).to(p.device())};
    }
  }
}
void Optimizer::reset() {
  at::NoGradGuard guard;
  for(size_t i=0;i<payload_.size();++i) {
    payload_[i].copy_(initial_[i]);states_[i].master.copy_(initial_[i]);
    states_[i].first.zero_();states_[i].second.zero_();states_[i].step.zero_();
  }
}
Tensor Optimizer::step(const Value& objective,const std::vector<Tensor>& gradients) {
  if(gradients.size()!=payload_.size())throw std::invalid_argument("bounded gradient inventory");
  at::NoGradGuard guard;auto used=objective.dependencies.any(0);
  std::vector<Tensor> values;auto finite=at::ones({},objective.data.options().dtype(at::kBool));
  for(size_t i=0;i<gradients.size();++i) {
    auto g=gradients[i].defined()?gradients[i].to(at::kFloat)/scale_:at::zeros_like(states_[i].master);
    auto mask=move(used[i],g.device());
    // Undefined is valid only for a structurally absent owner. Preserve this
    // check as a device predicate so capture never inspects the used mask.
    if(!gradients[i].defined())finite=finite & move(~mask,finite.device());
    finite=finite & move((~mask)|(at::isfinite(g).all() & (states_[i].step<max_updates_)),finite.device());values.push_back(g);
  }
  for(size_t i=0;i<states_.size();++i) {
    auto& s=states_[i];auto mask=move(used[i],s.master.device()) & move(finite,s.master.device());
    auto step=s.step+mask.to(at::kLong);auto g=values[i];Tensor first,second,next;
    if(kind_=="sgd") {
      g=g+.01*s.master;
      first=at::where(s.step==0,g,s.first*.9+g);second=s.second;
      next=s.master-lr_*first;
    } else {
      first=s.first.mul(.9).add(g,.1);second=s.second.mul(.999).addcmul(g,g,.001);
      const auto& table=corrections_.at(s.master.device().str());auto index=step.clamp(1,max_updates_).reshape({1});
      auto step_size=table.first.index_select(0,index).squeeze(0);
      auto correction2=table.second.index_select(0,index).squeeze(0);
      auto denominator=at::sqrt(second)/correction2+1e-5;
      next=s.master*(1-lr_*.01)-step_size*(first/denominator);
    }
    // Keep stable storage addresses for replay; update after backward consumed
    // this window's saved tensors. No in-place differentiable state mutation.
    s.first.copy_(at::where(mask,first,s.first));s.second.copy_(at::where(mask,second,s.second));
    s.step.copy_(step);s.master.copy_(at::where(mask,next,s.master));payload_[i].copy_(s.master);
  }
  for(const auto& p:payload_)finite=finite & move(at::isfinite(p).all(),finite.device());
  return finite;
}
std::vector<OptimizerState> Optimizer::snapshot() const {
  at::NoGradGuard guard;std::vector<OptimizerState> out;
  for(const auto& s:states_)out.push_back({s.master.clone(),s.first.clone(),s.second.clone(),s.step.clone()});
  return out;
}
}  // namespace accelerator_scale::bounded
