#pragma once
#include "tide/types.h"

namespace tide::device_online {
// Complete prefix lengths for cache rows; capacity==0 denotes a queue valid mask.
// Payloads are grouped tensors on one device, never a host list of messages.
struct ContinuationRows {std::vector<Tensor> values;Tensor lengths;Index capacity=0;};
struct SavedRows {std::vector<size_t> slots;Tensor indices;};
struct SavedBuffers {
  std::vector<Tensor> values;
  std::vector<std::vector<Index>> shapes;
  std::vector<SavedRows> groups;
  Index bytes=0;
};
SavedBuffers save_buffers(const std::vector<Tensor>&,const std::vector<ContinuationRows>&,Index budget);
void check_buffers(const std::vector<Tensor>&,const SavedBuffers&);
void restore_buffers(const std::vector<Tensor>&,const SavedBuffers&);
} // namespace tide::device_online
