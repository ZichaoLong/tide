#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>

namespace tide::device_online {
// Construction-only allocation plan. Every module's minimum is reserved before
// any module grows a physical chunk. All arithmetic precedes device allocation.
class ContentBudget {
 public:
  ContentBudget(int64_t total,bool aggressive,long double common,std::array<long double,8> minimum)
      :minimum_(minimum),remaining_(0),pending_(0) {
    if(total<1||common<0)throw std::invalid_argument("invalid content memory budget");
    for(auto x:minimum_){if(x<0)throw std::invalid_argument("invalid module minimum");pending_+=x;}
    const long double spare=total-common-pending_;
    if(spare<4096)throw std::invalid_argument("content module minima exceed memory budget");
    // Headroom is a share of what remains after mandatory persistent/minimum
    // work, not a reason to pretend required model bytes can shrink.
    const auto headroom=static_cast<int64_t>((spare-4096)*(aggressive?.1L:.25L));
    usable_=total-headroom;
    const auto operators=std::max<int64_t>(4096,static_cast<int64_t>(spare/8));
    remaining_=usable_-operators-common;
    if(remaining_<pending_)throw std::invalid_argument("content module minima exceed memory budget");
    common_=common;
  }
  int64_t available(size_t module) const {
    const auto active=std::count_if(minimum_.begin(),minimum_.end(),[](auto x){return x>0;});
    return static_cast<int64_t>(minimum_.at(module)+(remaining_-pending_)/std::max<decltype(active)>(1,active));
  }
  void reserve(size_t module,int64_t bytes) {
    if(bytes<minimum_.at(module)||bytes>available(module))throw std::logic_error("module violated its allocation budget");
    remaining_-=bytes;pending_-=minimum_[module];minimum_[module]=0;reserved_+=bytes;
  }
  int64_t planned_bytes() const {return static_cast<int64_t>(common_+reserved_);}
  int64_t operator_budget() const {return usable_-planned_bytes();}
  int64_t usable_bytes() const {return usable_;}
 private:
  int64_t usable_,reserved_=0;
  std::array<long double,8> minimum_;
  long double remaining_,pending_,common_=0;
};
} // namespace tide::device_online
