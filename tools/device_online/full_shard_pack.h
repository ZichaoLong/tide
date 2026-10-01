#pragma once
#include "broadcast_router.h"
namespace tide::device_online {
struct FullShardBatch {
  ActionBatch actions;
  at::Tensor content,comparison,destinations,branch;
};
FullShardBatch append_full_shard_pack(CannProgram&,const ActionBatch&,const at::Tensor& content,
    const at::Tensor& comparison,const at::Tensor& mapping,int64_t local_nodes,
    const at::Tensor& work,const at::Tensor& error);
} // namespace tide::device_online
