#pragma once
#include "full_vjp.h"
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

namespace tide::device_online {
// Internal, backward-group-scoped copies of immutable Full banks. The public
// owner forbids publication while windows are retained; version checks alone
// cannot detect CANN writes. Dynamic event records never enter this cache.
class RetainedFull {
 public:
  using Copies=std::map<const void*,at::Tensor>;
  static int64_t bytes(const FullTape& tape) {
    std::set<const void*> seen;long double size=0;
    for(const auto& x:values(tape))if(x.defined()&&seen.insert(x.unsafeGetTensorImpl()).second)size+=x.nbytes();
    if(size>std::numeric_limits<int64_t>::max())throw std::invalid_argument("retained Full extent overflow");
    return int64_t(size);
  }
  void validate(const FullTape& tape) const {
    if(!ready_)return;
    if(tape.samples!=samples_||tape.width!=width_||tape.has_tanh!=tanh_)changed();
    const auto current=values(tape);
    for(size_t i=0;i<current.size();++i)if(!sources_[i].matches(current[i]))changed();
  }
  int64_t reusable_bytes(const FullTape& tape) const {
    validate(tape);return ready_?bytes_:0;
  }
  void reuse(Copies& copies,const FullTape& tape) const {
    validate(tape);if(!ready_)return;
    for(size_t i=0;i<sources_.size();++i)if(sources_[i].tensor.defined())
      copies.emplace(sources_[i].tensor.unsafeGetTensorImpl(),copies_[i]);
  }
  // Call only after complete tape ownership and byte preflight. First capture
  // clones rather than borrowing writable forward banks, retaining aliases.
  void capture(const FullTape& tape) {
    validate(tape);if(ready_)return;
    const auto size=bytes(tape);Copies unique;
    std::vector<Source> sources;std::vector<at::Tensor> copies;
    for(const auto& x:values(tape)) {
      sources.emplace_back(x);
      if(!x.defined()){copies.emplace_back();continue;}
      auto& copy=unique[x.unsafeGetTensorImpl()];if(!copy.defined())copy=x.clone();copies.push_back(copy);
    }
    sources_=std::move(sources);copies_=std::move(copies);bytes_=size;
    samples_=tape.samples;width_=tape.width;tanh_=tape.has_tanh;ready_=true;
  }
  void seed(Copies& copies) const {
    if(ready_)for(size_t i=0;i<sources_.size();++i)if(sources_[i].tensor.defined())
      copies.at(sources_[i].tensor.unsafeGetTensorImpl())=copies_[i];
  }
  RetainedFull& shard(size_t index,size_t count) {
    if(shards_.empty())shards_.resize(count);
    if(shards_.size()!=count)throw std::logic_error("retained Full partition changed within backward group");
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
  static std::vector<at::Tensor> values(const FullTape& t) {
    return {t.kinds,t.weights,t.biases,t.extra.lh_kinds,t.extra.lh_weights,t.extra.lh_biases,
      t.extra.swiglu_kinds,t.extra.swiglu_mapping,t.extra.gate,t.extra.up,t.extra.down};
  }
  [[noreturn]] static void changed(){throw std::logic_error("retained Full banks changed within backward group");}
  bool ready_=false,tanh_=false;
  int64_t samples_=0,width_=0,bytes_=0;
  std::vector<Source> sources_;
  std::vector<at::Tensor> copies_;
  std::vector<RetainedFull> shards_;
};
} // namespace tide::device_online
