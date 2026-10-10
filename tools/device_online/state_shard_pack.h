#pragma once
#include "content_profile.h"

namespace tide::device_online {
struct StateShardBatch {
  // Complete owned fibers in original order, with compact node IDs. No local
  // frame selection is legal: region candidates may span multiple owners.
  // consumed/frame tables are absent; fibers[:,3] keeps the global frame ID.
  ReadyBatch ready;
  ContentBatch content;
  at::Tensor fiber_rows,atom_rows,destinations;
};
StateShardBatch append_state_shard_pack(DeviceProgram&,const ReadyBatch&,const ContentBatch&,
    const at::Tensor& mapping,int64_t local_nodes,int64_t capacity,const at::Tensor& error);
SelectionProposal append_state_shard_selection(DeviceProgram&,const StateShardBatch&,
    const SelectionProposal& global,const at::Tensor& error);
// source has capacity rows; destination has global rows + capacity distinct
// discard rows. No duplicate-index scatter, including inactive padding.
void append_state_shard_scatter(DeviceProgram&,const StateShardBatch&,const at::Tensor& source,
                               const at::Tensor& destination);
} // namespace tide::device_online
