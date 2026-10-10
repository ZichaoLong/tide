#pragma once
#include "broadcast_router.h"
#include "full_extra.h"
#include <string>

namespace tide::device_online {
// 0: not LH; 1..9: relu/silu/identity x identity/RMS/layer normalization.
int64_t lh_full_kind(const std::string&);
class PackedLhFull {
 public:
  static long double minimum_bytes(const std::vector<int64_t>& kinds,int64_t width,int64_t capacity,at::ScalarType dtype=at::kFloat);
  PackedLhFull(std::vector<int64_t> kinds,const at::Tensor& cpu_weight,const at::Tensor& cpu_bias,
               at::Device,int64_t capacity,int64_t max_chunk_rows,int64_t workspace_budget_bytes);
  ActionBatch append_stage(DeviceProgram&,const ActionBatch&,const at::Tensor& comparison,
                           const at::Tensor& error,const at::Tensor& chunks);
  int64_t chunk_rows() const {return chunk_;}
  int64_t reserved_bytes() const {return reserved_;}
  void tape(FullExtraTape& t) const {t.lh_kinds=kinds_;t.lh_weights=weights_;t.lh_biases=biases_;t.lh_groups=groups_;}
 private:
  int64_t nodes_,width_,rows_,chunk_,reserved_;
  std::vector<int64_t> groups_;
  at::Tensor kinds_,weights_,biases_;
};
} // namespace tide::device_online
