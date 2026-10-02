#pragma once
#include "fiber_tape.h"
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

namespace tide::device_online {
// Update-scoped immutable parameter copies. Fiber tape fields may be fresh
// gathers, so their identity is NOT a parameter-version guard: bind() checks
// the actual forward banks before each advance. The training owner also forbids
// publication while tapes are live (CANN writes need not increment _version).
// KV, lengths, log-bias and journals never enter this cache.
class RetainedAttention {
 public:
  using Copies=std::map<const void*,at::Tensor>;
  static int64_t bytes(const std::vector<EventAttentionTape>& event,const std::vector<FiberAttentionTape>& fiber) {
    std::set<const void*> seen;long double n=0;
    for(const auto& x:values(event,fiber))if(x.defined()&&seen.insert(x.unsafeGetTensorImpl()).second)n+=x.nbytes();
    if(n>std::numeric_limits<int64_t>::max())throw std::invalid_argument("retained attention extent overflow");
    return int64_t(n);
  }
  void bind(const std::vector<EventAttentionTape>& event,const FiberParameterBanks& fiber) {
    std::vector<at::Tensor> source;
    for(const auto& a:event){source.push_back(a.qkv);source.push_back(a.projection);}
    for(const auto& x:{fiber.qkv,fiber.qkv_bias,fiber.projection,fiber.projection_bias,fiber.decay,fiber.pool})source.push_back(x);
    if(bound_) {
      if(source.size()!=sources_.size())changed();
      for(size_t i=0;i<source.size();++i)if(!sources_[i].matches(source[i]))changed();
    } else {
      for(const auto& x:source)sources_.emplace_back(x);
      bound_=true;
    }
  }
  int64_t reusable_bytes(const std::vector<EventAttentionTape>& event,const std::vector<FiberAttentionTape>& fiber) const {
    if(!ready_)return 0;
    check(event,fiber);return bytes(event,fiber);
  }
  void reuse(Copies& copies,const std::vector<EventAttentionTape>& event,const std::vector<FiberAttentionTape>& fiber) const {
    if(!ready_)return;
    check(event,fiber);const auto current=values(event,fiber);
    for(size_t i=0;i<current.size();++i)if(current[i].defined())copies.emplace(current[i].unsafeGetTensorImpl(),copies_[i]);
  }
  // Called only after the complete tape's ownership and byte admission checks.
  void capture(const std::vector<EventAttentionTape>& event,const std::vector<FiberAttentionTape>& fiber) {
    if(ready_){check(event,fiber);return;}
    if(!bound_)throw std::logic_error("retained attention requires forward-bank binding");
    check_sources();const auto current=values(event,fiber);Copies unique;
    std::vector<at::Tensor> next;
    for(const auto& x:current) {
      if(!x.defined()){next.emplace_back();continue;}
      auto& copy=unique[x.unsafeGetTensorImpl()];if(!copy.defined())copy=x.clone();next.push_back(copy);
    }
    copies_=std::move(next);geometry_=geometry(event,fiber);aliases_=aliases(current);ready_=true;
  }
  void seed(Copies& copies,const std::vector<EventAttentionTape>& event,const std::vector<FiberAttentionTape>& fiber) const {
    check(event,fiber);const auto current=values(event,fiber);
    for(size_t i=0;i<current.size();++i)if(current[i].defined())copies.at(current[i].unsafeGetTensorImpl())=copies_[i];
  }
  RetainedAttention& shard(size_t index,size_t count) {
    if(shards_.empty())shards_.resize(count);
    if(shards_.size()!=count)throw std::logic_error("retained attention partition changed within update");
    return shards_.at(index);
  }
 private:
  struct Source {
    at::Tensor tensor;int64_t version=0;const void* data=nullptr;
    std::vector<int64_t> sizes,strides;
    at::ScalarType dtype=at::kFloat;at::Device device=at::kCPU;
    explicit Source(const at::Tensor& x):tensor(x) {
      if(x.defined()){version=x._version();data=x.const_data_ptr();sizes=x.sizes().vec();strides=x.strides().vec();dtype=x.scalar_type();device=x.device();}
    }
    bool matches(const at::Tensor& x) const {
      return x.defined()==tensor.defined()&&(!x.defined()||
        (x.unsafeGetTensorImpl()==tensor.unsafeGetTensorImpl()&&x._version()==version&&x.const_data_ptr()==data
          &&x.sizes()==at::IntArrayRef(sizes)&&x.strides()==at::IntArrayRef(strides)&&x.scalar_type()==dtype&&x.device()==device));
    }
  };
  static std::vector<at::Tensor> values(const std::vector<EventAttentionTape>& event,const std::vector<FiberAttentionTape>& fiber) {
    std::vector<at::Tensor> out;
    for(const auto& a:event){out.push_back(a.qkv);out.push_back(a.projection);}
    for(const auto& f:fiber)for(const auto& x:{f.cache.qkv,f.cache.projection,f.qkv_bias,f.projection_bias,f.decay,f.pool_weights})out.push_back(x);
    return out;
  }
  static std::vector<int64_t> aliases(const std::vector<at::Tensor>& tensors) {
    std::map<const void*,int64_t> seen;std::vector<int64_t> out;
    for(const auto& x:tensors) {
      if(!x.defined()){out.push_back(-1);continue;}
      out.push_back(seen.emplace(x.unsafeGetTensorImpl(),int64_t(seen.size())).first->second);
    }
    return out;
  }
  static std::vector<std::vector<int64_t>> geometry(const std::vector<EventAttentionTape>& event,const std::vector<FiberAttentionTape>& fiber) {
    std::vector<std::vector<int64_t>> out{{int64_t(event.size()),int64_t(fiber.size())}};
    auto add=[&](const EventAttentionTape& a){out.push_back(a.nodes);out.push_back({a.heads,a.kv_heads,a.width});};
    for(const auto& a:event)add(a);for(const auto& f:fiber)add(f.cache);return out;
  }
  [[noreturn]] static void changed(){throw std::logic_error("retained attention parameters changed within an update");}
  void check_sources() const {for(const auto& s:sources_)if(!s.matches(s.tensor))changed();}
  void check(const std::vector<EventAttentionTape>& event,const std::vector<FiberAttentionTape>& fiber) const {
    if(!bound_||!ready_)throw std::logic_error("retained attention snapshot is unavailable");
    check_sources();const auto current=values(event,fiber);
    if(geometry(event,fiber)!=geometry_||aliases(current)!=aliases_||current.size()!=copies_.size())changed();
    for(size_t i=0;i<current.size();++i)if(current[i].defined()!=copies_[i].defined()||
        (current[i].defined()&&(current[i].sizes()!=copies_[i].sizes()||current[i].scalar_type()!=copies_[i].scalar_type()
          ||current[i].device()!=copies_[i].device())))changed();
  }
  bool bound_=false,ready_=false;
  std::vector<Source> sources_;
  std::vector<at::Tensor> copies_;
  std::vector<int64_t> aliases_;
  std::vector<std::vector<int64_t>> geometry_;
  std::vector<RetainedAttention> shards_;
};
} // namespace tide::device_online
