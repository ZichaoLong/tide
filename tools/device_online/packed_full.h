#pragma once
#include "broadcast_router.h"

namespace tide::device_online {
// Actual selected actions only. Identity/tanh contracts share packed storage;
// chunks contain only selected tanh actions, plus safe zero sentinel rows.
// Metadata, cursor progression and chunk choice are device work.
class PackedFull {
 public:
  PackedFull(std::vector<int64_t> kinds,const at::Tensor& cpu_weight,const at::Tensor& cpu_bias,at::Device,
             int64_t max_chunk_rows,int64_t workspace_budget_bytes);
  ActionBatch append_stage(CannProgram&,const ActionBatch& content,const at::Tensor& comparison,
                           const at::Tensor& error);
  const at::Tensor& chunks() const {return chunks_;}
  int64_t chunk_rows() const {return chunk_;}
  static long double minimum_bytes(const std::vector<int64_t>& kinds,int64_t width);
 private:
  int64_t nodes_,width_,chunk_;
  bool any_tanh_;
  at::Tensor kinds_,weights_,biases_,chunks_;
};
} // namespace tide::device_online
