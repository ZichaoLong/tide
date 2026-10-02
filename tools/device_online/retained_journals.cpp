#include "retained_journals.h"
#include <ATen/core/grad_mode.h>
#include <algorithm>
#include <cstdint>
#include <map>
#include <stdexcept>

namespace tide::device_online {
namespace {
class Prefixes {
 public:
  int64_t rows(const at::Tensor& count,int64_t capacity) {
    if(at::GradMode::is_enabled()||capacity<1||count.sizes()!=at::IntArrayRef{1}||count.scalar_type()!=at::kLong
        ||count.device().type()!=c10::DeviceType::PrivateUse1)
      throw std::invalid_argument("invalid retained journal count/capacity");
    const auto key=std::make_pair(reinterpret_cast<std::uintptr_t>(count.unsafeGetTensorImpl()),capacity);
    auto found=extents_.find(key);if(found!=extents_.end())return found->second;
    // This dynamic output extent synchronizes once at the completed-window
    // retention boundary, not once per event. No numerical record is exported.
    const auto index=at::arange(capacity,count.options());
    const auto extent=at::nonzero(index<count).numel();
    const auto n=std::max<int64_t>(1,extent);extents_.emplace(key,n);return n;
  }
  void prefix(at::Tensor& value,int64_t n) {
    if(!value.defined())return;
    if(value.dim()<1||value.size(0)<n)throw std::invalid_argument("retained journal prefix exceeds tensor");
    const auto key=std::make_pair(reinterpret_cast<std::uintptr_t>(value.unsafeGetTensorImpl()),n);
    auto found=views_.find(key);
    if(found==views_.end())found=views_.emplace(key,value.narrow(0,0,n)).first;
    value=found->second; // Preserve aliases so retention still clones only once.
  }
  void journal(at::Tensor& metadata,at::Tensor& values,const at::Tensor& count) {
    if(!metadata.defined())return;
    const auto n=rows(count,metadata.size(0));prefix(metadata,n);prefix(values,n);
  }
  void cache(EventAttentionTape& tape) {journal(tape.metadata,tape.values,tape.count);}
 private:
  using Key=std::pair<std::uintptr_t,int64_t>;
  std::map<Key,int64_t> extents_;
  std::map<Key,at::Tensor> views_;
};
}
void compact_retained_journals(ReverseTape& t) {
  Prefixes p;const auto events=p.rows(t.state.count,t.state.metadata.size(0));
  p.prefix(t.state.metadata,events);p.prefix(t.state.values,events);
  p.prefix(t.full.metadata,events);p.prefix(t.full.values,events);
  p.prefix(t.full_values,events);p.prefix(t.control.raw_full,events);
  p.journal(t.fiber_meta,t.fiber_values,t.fiber_count);
  p.journal(t.emission.metadata,t.emission.values,t.emission.count);
  for(auto& a:t.attention)p.cache(a);
  for(auto& f:t.fiber)p.cache(f.cache);
}
void compact_retained_journals(StateOwnerTape& t) {
  Prefixes p;for(auto& a:t.attention)p.cache(a);
  for(auto& f:t.fiber)p.cache(f.cache);
}
} // namespace tide::device_online
