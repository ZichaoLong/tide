#pragma once
#include "cann_program.h"
#include "packed_queue.h"

namespace tide::device_online {
// Bounded, mutable inference queue. Build one device transaction per stage;
// replay uses current metadata and payloads, with no host count/branch reads.
// All producers must finish before this stage. Do not alias incoming with the
// transaction's private scratch. Errors are sticky; the old queue is preserved.
class QueueTransaction {
 public:
  QueueTransaction(int64_t capacity,int64_t width,int64_t nodes,int64_t samples,
                   at::TensorOptions);
  const AtomBatch& atoms() const {return atoms_;}
  const at::Tensor& error() const {return error_;} // 1 capacity, 2 invalid coordinates
  const at::Tensor& stats() const {return stats_;} // [current live count, peak]
  // consumed: int32[capacity], exactly 0/1. Complete-fiber selection is the
  // scheduler's obligation. This primitive preserves survivor/arrival order.
  void append_stage(CannProgram&,const at::Tensor& consumed,const AtomBatch& incoming);
 private:
  int64_t capacity_,width_,nodes_,samples_;
  AtomBatch atoms_;
  at::Tensor error_,stats_;
};
} // namespace tide::device_online
