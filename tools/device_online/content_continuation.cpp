#include "content_flow_internal.h"
#include <ATen/core/grad_mode.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <limits>
#include <set>
#include <stdexcept>

namespace tide::device_online {
struct SavedContent {
  std::shared_ptr<const int> owner;
  Continuation boundary; // Only CPU identity, input ledger, batch size and cut.
  std::vector<Tensor> tensors;
  Index bytes=0;
};
namespace {
void synchronize(const std::vector<Tensor>& tensors) {
  std::set<std::pair<c10::DeviceType,int>> devices;
  for(const auto& x:tensors)devices.insert({x.device().type(),x.device().index()});
  for(const auto& [type,index]:devices)c10::impl::VirtualGuardImpl(type).synchronizeDevice(index);
}
}
std::vector<Tensor> ContentFlow::Impl::continuation_tensors() const {
  auto q=pending->atoms();
  std::vector<Tensor> out{history.counts,history.seen,history.last_time,history.present,
    q.coordinates,q.values,q.valid,pending->stats()};
  auto append=[&](std::vector<Tensor> xs){out.insert(out.end(),xs.begin(),xs.end());};
  if(sharded_state)append(sharded_state->continuation_tensors());
  else {
    append({state.values,state.clocks,state.present});
    if(attention)append(attention->continuation_tensors());
    if(event_attention)append(event_attention->continuation_tensors());
  }
  return out;
}
void ContentFlow::Impl::reset_window() {
  outputs->atoms().valid.zero_();outputs->stats().zero_();
  if(limits.diagnostics) {
    if(export_diagnostics){messages->atoms().valid.zero_();messages->stats().zero_();contributions->count.zero_();}
    events->count.zero_();fibers->count.zero_();full_trace->count.zero_();emission_trace->count.zero_();
    if(raw_full_trace)raw_full_trace->count.zero_();
  }
  if(attention)attention->reset_window();
  if(event_attention)event_attention->reset_window();
  if(aggregate)aggregate->chunks().zero_();
  event_count.zero_();if(full)full->chunks().zero_();if(remote_full)full_chunks.zero_();
  if(sharded_full)sharded_full->reset_window();
  if(sharded_state)sharded_state->reset_window();
  emission->chunks().zero_();stages.zero_();
}
ResidentContinuation ContentFlow::snapshot_device(Index max_bytes) const {
  if(!impl_||impl_->failed)throw std::logic_error("device continuation unavailable on closed/failed flow");
  if(at::GradMode::is_enabled())throw std::invalid_argument("device continuation requires no-grad");
  auto& s=*impl_;const auto buffers=s.continuation_tensors();long double bytes=0;
  for(const auto& x:buffers)bytes+=x.nbytes();
  if(max_bytes<1||bytes>max_bytes)throw std::invalid_argument("device continuation tensor budget exceeded");
  auto saved=std::make_shared<SavedContent>();saved->owner=s.continuation_owner;
  saved->boundary=s.boundary;saved->bytes=static_cast<Index>(bytes);
  // Copy whole buffers/groups on their current devices. Never download state,
  // KV, history, pending coordinates/counts or select individual messages here.
  try {
    synchronize(buffers);
    for(const auto& x:buffers)saved->tensors.push_back(x.clone());
    synchronize(saved->tensors);
  }catch(...){s.failed=true;throw;}
  ResidentContinuation out;out.data_=std::move(saved);return out;
}
void ContentFlow::restore_device(const ResidentContinuation& saved) {
  if(!impl_||impl_->failed)throw std::logic_error("device continuation unavailable on closed/failed flow");
  if(at::GradMode::is_enabled())throw std::invalid_argument("device continuation requires no-grad");
  auto& s=*impl_;
  if(!saved.data_||saved.data_->owner!=s.continuation_owner)
    throw std::invalid_argument("device continuation belongs to a different session");
  const auto& value=*saved.data_;const auto buffers=s.continuation_tensors();
  if(buffers.size()!=value.tensors.size())throw std::logic_error("device continuation layout changed");
  for(size_t i=0;i<buffers.size();++i) {
    const auto& a=buffers[i];const auto& b=value.tensors[i];
    if(a.device()!=b.device()||a.scalar_type()!=b.scalar_type()||a.sizes()!=b.sizes())
      throw std::logic_error("device continuation tensor layout changed");
  }
  // A rejected ownership/layout check cannot change live state. Runtime failure
  // during copies poisons this flow, as with a failed forward submission.
  try {
    synchronize(buffers);
    for(size_t i=0;i<buffers.size();++i)buffers[i].copy_(value.tensors[i]);
    s.reset_window();s.external.valid.zero_();s.stop.fill_(value.boundary.cut);
    synchronize(buffers);
    s.boundary=value.boundary;s.window_start=s.boundary.cut;
  }catch(...){s.failed=true;throw;}
}
} // namespace tide::device_online

namespace tide {
Index ResidentContinuation::cut() const {
  if(!data_)throw std::logic_error("empty device continuation");return data_->boundary.cut;
}
Index ResidentContinuation::batch_size() const {
  if(!data_)throw std::logic_error("empty device continuation");return data_->boundary.batch_size;
}
Index ResidentContinuation::tensor_bytes() const {
  if(!data_)throw std::logic_error("empty device continuation");return data_->bytes;
}
} // namespace tide
