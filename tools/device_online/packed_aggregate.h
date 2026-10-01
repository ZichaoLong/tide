#pragma once
#include "content_profile.h"
#include "packed_sum.h"
#include "aggregate_tape.h"

namespace tide::device_online {
int64_t aggregate_kind(const std::string&);
int64_t aggregate_slots(const Graph&);
// Optional normalized Aggregate stage. Source parameters are packed statically;
// present domains and chunk membership are generated from actual ready fibers.
// Physical scales precede normalization; contributions preserve physical order.
class PackedAggregate {
 public:
  static long double minimum_bytes(const ContentProfile&,int64_t rows);
  PackedAggregate(const ContentProfile&,at::Device,int64_t rows,int64_t chunk,int64_t budget);
  void append(CannProgram&,const ReadyBatch&,const PackedSum&,const at::Tensor& error,bool vectorized) const;
  int64_t reserved_bytes() const {return reserved_;}
  int64_t chunk_rows() const {return chunk_;}
  at::Tensor chunks() const {return chunks_;}
  AggregateTape tape() const {return {kinds_,lengths_,weights_,slots_};}
 private:
  int64_t rows_,chunk_,slots_,reserved_;
  at::ScalarType dtype_;
  at::Tensor kinds_,lengths_,weights_,sources_,chunks_;
};
} // namespace tide::device_online
