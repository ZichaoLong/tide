#pragma once
#include <ATen/ATen.h>
#include <array>
#include <map>
#include <limits>
#include <stdexcept>

namespace tide::device_online {
// Internal update-scoped immutable copies, never borrowed forward banks. Public
// training owners prohibit parameter publication while windows are live and
// discard this cache after backward/detach/close. CANN writes do not necessarily
// increment ATen versions: this lifecycle, not _version alone, is essential.
class RetainedProjection {
 public:
  static int64_t bytes(const at::Tensor& weights,const at::Tensor& biases) {
    if(weights.defined()!=biases.defined())throw std::invalid_argument("incomplete retained projection banks");
    if(!weights.defined())return 0;
    const long double size=static_cast<long double>(weights.nbytes())+
      (weights.unsafeGetTensorImpl()==biases.unsafeGetTensorImpl()?0.L:biases.nbytes());
    if(size>std::numeric_limits<int64_t>::max())throw std::invalid_argument("retained projection extent overflow");
    return int64_t(size);
  }
  int64_t reusable_bytes(const at::Tensor& weights,const at::Tensor& biases) const {
    if(!ready_)return 0;
    check(weights,biases);return bytes_;
  }
  void reuse(std::map<const void*,at::Tensor>& copies,const at::Tensor& weights,const at::Tensor& biases) const {
    if(!ready_)return;
    check(weights,biases);
    for(size_t i=0;i<2;++i)copies.emplace(source_[i].unsafeGetTensorImpl(),copies_[i]);
  }
  void capture(const at::Tensor& weights,const at::Tensor& biases) {
    if(ready_){check(weights,biases);return;}
    bytes_=bytes(weights,biases);if(!bytes_)return;
    source_={weights,biases};
    for(size_t i=0;i<2;++i) {
      versions_[i]=source_[i]._version();data_[i]=source_[i].const_data_ptr();
      copies_[i]=i&&source_[i].unsafeGetTensorImpl()==source_[0].unsafeGetTensorImpl()?copies_[0]:source_[i].clone();
    }
    ready_=true;
  }
  // Called after shape/ownership/budget preflight, before per-window copies.
  void seed(std::map<const void*,at::Tensor>& copies) const {
    if(ready_)for(size_t i=0;i<2;++i)copies.at(source_[i].unsafeGetTensorImpl())=copies_[i];
  }
 private:
  void check(const at::Tensor& weights,const at::Tensor& biases) const {
    const std::array<at::Tensor,2> current{weights,biases};
    for(size_t i=0;i<2;++i)
      if(!current[i].defined()||current[i].unsafeGetTensorImpl()!=source_[i].unsafeGetTensorImpl()
          ||current[i]._version()!=versions_[i]||current[i].const_data_ptr()!=data_[i]
          ||current[i].sizes()!=copies_[i].sizes()||current[i].scalar_type()!=copies_[i].scalar_type()
          ||current[i].device()!=copies_[i].device())
        throw std::logic_error("retained projection changed within an update");
  }
  bool ready_=false;
  int64_t bytes_=0;
  std::array<at::Tensor,2> source_,copies_;
  std::array<int64_t,2> versions_{};
  std::array<const void*,2> data_{};
};
} // namespace tide::device_online
