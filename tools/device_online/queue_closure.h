#pragma once
#include "packed_queue.h"
#include <tuple>
#include <vector>

namespace tide::device_online {
using Wire = std::tuple<int64_t,int64_t,int64_t>; // source node,target node,delay
class QueueClosure {
 public:
  QueueClosure(const std::vector<int64_t>& node_regions, int64_t regions,
               const std::vector<Wire>& wires, int64_t samples, at::Device,
               int64_t workspace_budget_bytes=64*1024*1024);
  // stop is exact int64 [1], complete input seal. Only metadata operations;
  // caller must reject invalid coordinates before numerical/index dispatch.
  at::Tensor ready(const AtomBatch&, const at::Tensor& stop, bool prefill) const;
  at::Tensor valid_coordinates(const AtomBatch&) const;
  int64_t sample_chunk() const { return sample_chunk_; }
  const at::Tensor& owners() const { return owner_; }
  const at::Tensor& distances() const { return distance_; }
 private:
  int64_t regions_, samples_, nodes_, sample_chunk_;
  at::Tensor owner_, distance_;
};
}  // namespace tide::device_online
