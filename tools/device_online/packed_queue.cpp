#include "packed_queue.h"
#include <ATen/core/grad_mode.h>
#include <limits>
#include <stdexcept>

namespace tide::device_online {
PackedQueue::PackedQueue(int64_t capacity, int64_t width, at::TensorOptions opts)
    : capacity_(capacity), width_(width) {
  if (capacity<1 || width<1 || width>std::numeric_limits<int64_t>::max()/capacity/8)
    throw std::invalid_argument("invalid packed queue capacity/width");
  if (at::GradMode::is_enabled()) throw std::invalid_argument("mutable packed queue requires explicit no-grad mode");
  atoms_={at::zeros({capacity,6},opts.dtype(at::kLong)), at::zeros({capacity,width},opts),
          at::zeros({capacity},opts.dtype(at::kBool))};
  error_=at::zeros({1},opts.dtype(at::kInt)); peak_=at::zeros({1},opts.dtype(at::kLong));
}
void PackedQueue::validate(const AtomBatch& in) const {
  if (!in.coordinates.defined() || !in.values.defined() || !in.valid.defined()
      || in.coordinates.dim()!=2 || in.coordinates.size(1)!=6 || in.coordinates.size(0)<1
      || in.values.sizes()!=at::IntArrayRef{in.coordinates.size(0),width_}
      || in.valid.sizes()!=at::IntArrayRef{in.coordinates.size(0)}
      || in.coordinates.scalar_type()!=at::kLong || in.valid.scalar_type()!=at::kBool
      || in.values.scalar_type()!=atoms_.values.scalar_type())
    throw std::invalid_argument("invalid packed atom buffers");
  for (const auto& t : {in.coordinates,in.values,in.valid})
    if (t.device()!=atoms_.valid.device() || t.requires_grad())
      throw std::invalid_argument("packed atoms require matching device and no autograd");
}
void PackedQueue::validate_mask(const at::Tensor& mask) const {
  if (!mask.defined() || mask.sizes()!=atoms_.valid.sizes() || mask.scalar_type()!=at::kBool
      || mask.device()!=atoms_.valid.device()) throw std::invalid_argument("invalid packed queue mask");
}
void PackedQueue::append(const AtomBatch& in) {
  replace(at::zeros_like(atoms_.valid),in);
}
void PackedQueue::replace(const at::Tensor& consumed,const AtomBatch& in) {
  validate_mask(consumed);
  validate(in);
  auto valid=at::cat({atoms_.valid&~consumed,in.valid});
  auto count=valid.sum(at::kLong).reshape({1});
  auto overflow=count>capacity_, accept=(error_==0)&~overflow;
  // A stable valid-first compaction preserves physical identities and tie order.
  // Retain the original queue on failure, including its holes and payload bits.
  auto order=at::argsort(valid.to(at::kLong),true,0,true).slice(0,0,capacity_);
  auto coords=at::cat({atoms_.coordinates,in.coordinates}).index_select(0,order);
  auto values=at::cat({atoms_.values,in.values}).index_select(0,order);
  atoms_.coordinates.copy_(at::where(accept,coords,atoms_.coordinates));
  atoms_.values.copy_(at::where(accept,values,atoms_.values));
  atoms_.valid.copy_(at::where(accept,valid.index_select(0,order),atoms_.valid));
  peak_.copy_(at::where(accept,at::maximum(peak_,count),peak_));
  error_.copy_(at::where((error_==0)&overflow,at::ones_like(error_),error_));
}
void PackedQueue::erase(const at::Tensor& mask) {
  validate_mask(mask);
  atoms_.valid.logical_and_(~mask | (error_!=0));
}
AtomBatch PackedQueue::pack(const at::Tensor& mask) const {
  validate_mask(mask);
  auto valid=atoms_.valid & mask;
  auto order=at::arange(capacity_,atoms_.coordinates.options());
  // Lexicographic stable sorting, no lossy integer composite key.
  for (int field=5;field>=0;--field) {
    auto keys=atoms_.coordinates.select(1,field).index_select(0,order);
    order=order.index_select(0,at::argsort(keys,true,0,false));
  }
  order=order.index_select(0,at::argsort(valid.index_select(0,order).to(at::kLong),true,0,true));
  return {atoms_.coordinates.index_select(0,order),atoms_.values.index_select(0,order),valid.index_select(0,order)};
}
void PackedQueue::clear() { atoms_.valid.zero_(); error_.zero_(); peak_.zero_(); }
}  // namespace tide::device_online
