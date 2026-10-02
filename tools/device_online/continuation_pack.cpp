#include "continuation_pack.h"
#include <map>
#include <set>
#include <stdexcept>

namespace tide::device_online {
SavedBuffers save_buffers(const std::vector<Tensor>& buffers,const std::vector<ContinuationRows>& groups,Index budget) {
  if(budget<1)throw std::invalid_argument("device continuation tensor budget exceeded");
  SavedBuffers out;out.values.resize(buffers.size());std::map<const void*,size_t> slots;
  long double bytes=0;std::set<size_t> packed;
  for(size_t i=0;i<buffers.size();++i) {
    slots.emplace(buffers[i].unsafeGetTensorImpl(),i);bytes+=buffers[i].nbytes();
    out.shapes.push_back(buffers[i].sizes().vec());
  }
  for(const auto& group:groups) {
    // nonzero keeps all valid rows, including numerically zero payloads. Its
    // dynamic output extent synchronizes at this explicit detached save boundary;
    // no indices/length vectors/payloads are materialized on CPU.
    Tensor valid=group.lengths;
    if(group.capacity)valid=at::arange(group.capacity,group.lengths.options()).unsqueeze(0)
                                <group.lengths.unsqueeze(1);
    auto indices=at::nonzero(valid.reshape({-1})).reshape({-1});
    SavedRows saved;long double full=0,compact=indices.nbytes();
    for(const auto& value:group.values) {
      const auto slot=slots.at(value.unsafeGetTensorImpl());
      if(packed.count(slot))throw std::logic_error("overlapping continuation row groups");
      saved.slots.push_back(slot);full+=value.nbytes();
      compact+=static_cast<long double>(indices.numel())*(value.nbytes()/value.size(0));
    }
    if(compact>=full)continue; // Dense caches need no additional index storage.
    bytes+=compact-full;saved.indices=std::move(indices);
    for(auto slot:saved.slots)packed.insert(slot);
    out.groups.push_back(std::move(saved));
  }
  // Decide capacity after device row selection, before allocating payload copies.
  if(bytes>budget)throw std::invalid_argument("device continuation tensor budget exceeded");
  out.bytes=static_cast<Index>(bytes);
  for(const auto& group:out.groups)for(auto slot:group.slots)
    out.values[slot]=buffers[slot].index_select(0,group.indices);
  for(size_t i=0;i<buffers.size();++i)if(!packed.count(i))out.values[i]=buffers[i].clone();
  return out;
}
void check_buffers(const std::vector<Tensor>& buffers,const SavedBuffers& saved) {
  if(buffers.size()!=saved.values.size()||buffers.size()!=saved.shapes.size())
    throw std::logic_error("device continuation layout changed");
  for(size_t i=0;i<buffers.size();++i)
    if(buffers[i].sizes()!=saved.shapes[i]||buffers[i].device()!=saved.values[i].device()
        ||buffers[i].scalar_type()!=saved.values[i].scalar_type())
      throw std::logic_error("device continuation tensor layout changed");
}
void restore_buffers(const std::vector<Tensor>& buffers,const SavedBuffers& saved) {
  std::set<size_t> packed;
  for(const auto& group:saved.groups)for(auto slot:group.slots) {
    // Unique original row indices preserve physical identities and stable order.
    // Padding and independent cache sentinels are reset, never used as operands.
    buffers[slot].zero_();
    if(group.indices.numel())buffers[slot].index_copy_(0,group.indices,saved.values[slot]);
    packed.insert(slot);
  }
  for(size_t i=0;i<buffers.size();++i)if(!packed.count(i))buffers[i].copy_(saved.values[i]);
}
} // namespace tide::device_online
