#pragma once
#include "capacity.h"
#include <tide/resident.h>

namespace tide_flow {
using tide::Index;
class ContextPool {
 public:
  ContextPool(Index count,const std::vector<at::Device>& devices,const capacity::Plan& plan) {
    if(count>1)handles_.resize(count);
    for(size_t i=0;i<devices.size();++i) {
      auto d=devices[i].index();budgets_[d]=plan.cards[i].components.at("saved_contexts");used_[d]=peak_[d]=0;
    }
  }
  bool empty() const {return handles_.empty();}
  const tide::ResidentContinuation& operator[](size_t i) const {return handles_.at(i);}
  std::map<Index,Index> remaining() const {
    auto out=budgets_;for(auto& [d,n]:out)n-=used_.at(d);return out;
  }
  void initialize(const tide::ResidentContinuation& value) {
    for(size_t i=0;i<handles_.size();++i)store(i,value);
  }
  void release(size_t i) {
    for(const auto& [d,n]:handles_.at(i).device_bytes())used_.at(d)-=n;
    handles_[i]={};
  }
  void store(size_t i,const tide::ResidentContinuation& value) {
    const auto sizes=value.device_bytes();
    for(const auto& [d,n]:sizes)if(n>budgets_.at(d)-used_.at(d))
      throw std::invalid_argument("saved continuation pool budget exceeded");
    handles_.at(i)=value;
    for(const auto& [d,n]:sizes){used_.at(d)+=n;peak_.at(d)=std::max(peak_.at(d),used_.at(d));}
  }
  const std::map<Index,Index>& peaks() const {return peak_;}
 private:
  std::vector<tide::ResidentContinuation> handles_;
  std::map<Index,Index> budgets_,used_,peak_;
};
} // namespace tide_flow
