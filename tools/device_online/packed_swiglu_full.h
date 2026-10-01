#pragma once
#include "broadcast_router.h"
#include "full_extra.h"
#include "tide/types.h"

namespace tide::device_online {
struct ContentProfile;
class PackedSwiGluFull {
 public:
  static long double minimum_bytes(const ContentProfile&,int64_t capacity);
  static long double minimum_bytes(const std::vector<Node>&,int64_t width,at::ScalarType,int64_t capacity);
  PackedSwiGluFull(const ContentProfile&,at::Device,int64_t capacity,int64_t max_rows,int64_t budget);
  PackedSwiGluFull(const std::vector<Node>&,const std::vector<NodeWeights>&,int64_t width,at::ScalarType,
                  at::Device,int64_t capacity,int64_t max_rows,int64_t budget);
  ActionBatch append_stage(CannProgram&,const ActionBatch&,const at::Tensor& content,
      const at::Tensor& comparison,const at::Tensor& error,const at::Tensor& chunks);
  int64_t chunk_rows() const {return chunk_;}
  int64_t reserved_bytes() const {return reserved_;}
  void tape(FullExtraTape& t) const {t.swiglu_kinds=kinds_;t.swiglu_mapping=mapping_;t.gate=gate_;t.up=up_;t.down=down_;}
 private:
  int64_t nodes_,width_,rows_,parameters_,chunk_,reserved_;
  at::Tensor kinds_,mapping_,gate_,up_,down_;
};
} // namespace tide::device_online
