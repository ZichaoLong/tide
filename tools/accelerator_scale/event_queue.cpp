#include "dispatch.h"
#include <algorithm>

namespace accelerator_scale {
TensorEventQueue::TensorEventQueue(at::Device device):device_(device) {
  keys_=at::empty({0,6},at::TensorOptions().dtype(at::kLong).device(device));
}
void TensorEventQueue::push(const Atom& atom) { staged_.push_back(atom); }
void TensorEventQueue::flush() {
  if(staged_.empty())return;
  at::NoGradGuard guard;std::vector<Index> data;data.reserve(staged_.size()*6);
  for(const auto& a:staged_)data.insert(data.end(),{a.batch,a.node,a.time,a.kind,a.source,a.position});
  auto incoming=at::from_blob(data.data(),{static_cast<Index>(staged_.size()),6},at::TensorOptions().dtype(at::kLong)).clone().to(device_);
  keys_=at::cat({keys_,incoming},0);
  if(!device_.is_cpu())transfers.metadata_host_to_device+=data.size()*sizeof(Index);
  pending_.insert(pending_.end(),std::make_move_iterator(staged_.begin()),std::make_move_iterator(staged_.end()));
  staged_.clear();
}
bool TensorEventQueue::pop(Index stop,std::vector<Atom>& arrived) {
  flush();if(pending_.empty())return false;
  at::NoGradGuard guard;
  const auto times=keys_.select(1,2);const auto next=times.min().item<Index>();
  if(!device_.is_cpu())transfers.metadata_device_to_host+=sizeof(Index);
  if(next>=stop)return false;
  auto ids=at::nonzero(times==next).flatten();auto subset=keys_.index_select(0,ids);
  auto order=at::arange(ids.numel(),ids.options());
  // Sorting is on the selected device; host reconstruction follows returned
  // indices and never re-sorts or compares scheduling keys.
  for(Index column=5;column>=0;--column) {
    if(column==2)continue;
    order=order.index_select(0,at::argsort(subset.select(1,column).index_select(0,order),true,0,false));
  }
  auto host_ids=ids.index_select(0,order).to(at::kCPU).contiguous();
  if(!device_.is_cpu())transfers.metadata_device_to_host+=host_ids.numel()*sizeof(Index);
  const auto* values=host_ids.data_ptr<Index>();std::vector<bool> used(pending_.size(),false);
  arrived.clear();arrived.reserve(ids.numel());
  for(Index i=0;i<host_ids.numel();++i){used[values[i]]=true;arrived.push_back(std::move(pending_[values[i]]));}
  std::vector<Atom> rest;rest.reserve(pending_.size()-arrived.size());
  for(size_t i=0;i<pending_.size();++i)if(!used[i])rest.push_back(std::move(pending_[i]));
  pending_=std::move(rest);keys_=keys_.index_select(0,at::nonzero(times!=next).flatten());
  return true;
}
void TensorEventQueue::export_pending(Continuation& state) const {
  state.pending=pending_;state.pending.insert(state.pending.end(),staged_.begin(),staged_.end());
  std::sort(state.pending.begin(),state.pending.end(),[](const Atom& a,const Atom& b){return a.key()<b.key();});
}
}  // namespace accelerator_scale
