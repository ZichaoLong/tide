#pragma once
#include "broadcast_router.h"
#include "full_placement.h"
#include "projection_shard.h"

namespace tide::device_online {
struct ContentProfile;
struct EmissionBatch {
  // Actual present slots: sample,node,time,position,local slot,parameter row.
  // Parameter row -1 denotes broadcast; physical delivery identity is separate.
  at::Tensor meta,values,count;
  AtomBatch arrivals,outputs;
};
// HARD FP32/FP16 emission, after Full. Static tables describe local slots only;
// presence and selected projection chunks are decided inside the device loop.
class PackedEmission {
 public:
  static long double minimum_bytes(const ContentProfile&,int64_t arrivals,int64_t outputs,const FullPlacement& placement={});
  PackedEmission(const ContentProfile&,at::Device,int64_t samples,int64_t arrivals,
                 int64_t outputs,int64_t max_chunk_rows,int64_t budget,const FullPlacement& placement={});
  EmissionBatch append_stage(DeviceProgram&,const ActionBatch&,const at::Tensor& error,int64_t operator_budget=0);
  int64_t reserved_bytes() const {return reserved_;}
  int64_t chunk_rows() const {return chunk_;}
  const at::Tensor& chunks() const {return chunks_;}
  const at::Tensor& scales() const {return scales_;}
  const at::Tensor& weights() const {return weights_;}
  const at::Tensor& biases() const {return biases_;}
  const std::vector<ProjectionBank>& shards() const {return shards_;}
  int64_t program_count() const;
  ProjectionStage* peer() const {return projection_.get();}
 private:
  int64_t nodes_,samples_,width_,arrivals_,outputs_,capacity_,slots_,parameters_,chunk_,reserved_;
  at::Tensor offsets_,periods_,slots_table_,scales_,weights_,biases_,chunks_;
  int64_t projection_budget_=0;
  std::vector<ProjectionBank> shards_;
  std::unique_ptr<ProjectionStage> projection_;
};
} // namespace tide::device_online
