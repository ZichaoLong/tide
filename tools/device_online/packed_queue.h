#pragma once
#include <ATen/ATen.h>

namespace tide::device_online {
// Actual messages only, never a potential-event table. Coordinates are
// (sample,node,time,kind,physical source,position); valid is independent of value.
struct AtomBatch { at::Tensor coordinates, values, valid; };
class PackedQueue {
 public:
  PackedQueue(int64_t capacity, int64_t width, at::TensorOptions);
  const AtomBatch& atoms() const { return atoms_; }
  at::Tensor error() const { return error_; } // int32 [1]: 0 OK, 1 capacity
  at::Tensor size() const { return atoms_.valid.sum(at::kLong).reshape({1}); }
  at::Tensor peak() const { return peak_; }
  // Fixed-shape bulk operations, no tensor->host decisions. append is atomic
  // on overflow: preserve the old queue and set sticky error, never truncate.
  void append(const AtomBatch&);
  void replace(const at::Tensor& consumed, const AtomBatch& incoming);
  void erase(const at::Tensor& mask);
  AtomBatch pack(const at::Tensor& mask) const;
  void clear(); // Explicit caller boundary; not an implicit continuation reset.
 private:
  int64_t capacity_, width_;
  AtomBatch atoms_;
  at::Tensor error_, peak_;
  void validate(const AtomBatch&) const;
  void validate_mask(const at::Tensor&) const;
};
}  // namespace tide::device_online
