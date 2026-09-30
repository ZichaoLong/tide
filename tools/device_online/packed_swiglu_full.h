#pragma once
#include "broadcast_router.h"

namespace tide::device_online {
struct ContentProfile;
class PackedSwiGluFull {
 public:
  static long double minimum_bytes(const ContentProfile&,int64_t capacity);
  PackedSwiGluFull(const ContentProfile&,at::Device,int64_t capacity,int64_t max_rows,int64_t budget);
  ActionBatch append_stage(CannProgram&,const ActionBatch&,const at::Tensor& content,
      const at::Tensor& comparison,const at::Tensor& error,const at::Tensor& chunks);
  int64_t chunk_rows() const {return chunk_;}
  int64_t reserved_bytes() const {return reserved_;}
 private:
  int64_t nodes_,width_,rows_,parameters_,chunk_,reserved_;
  at::Tensor kinds_,mapping_,gate_,up_,down_;
};
} // namespace tide::device_online
