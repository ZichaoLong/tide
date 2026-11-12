#pragma once
// Fixed KV §1.3/§3: whole B x L x N reception sequence with exact raw states.
// A sequence is a numerical dependency boundary, not a host event schedule.
#include "weighted_kv_update.h"
#include "weighted_row_graph.h"

namespace tide::device_online {
struct BlockKVSequenceTape {
  weighted::KVConfig config;
  int64_t batch=0,length=0,nodes=0;
  WeightedRawLayout layout;
  WeightedKVUpdateTape writes; // One independently projected positive write per row.
  WeightedRawState old,proposal,final;
  at::Tensor fields,mapping,initial_integer;
};
struct BlockKVSequenceRoots {
  at::Tensor old,old_connected,proposal,proposal_connected,final,final_connected;
};
struct BlockKVSequenceGradient {
  at::Tensor initial,initial_connected;
  WeightedKVUpdateGradient update; // content [B*L*N,d+1] and parameter roles.
};
// initial [B*N,*], content [B*L*N,d+1], present/times/owners [B,L,N].
// FP32 raw; capacity bounds simultaneously retained entries, not total time.
// The caller owns state publication. Returned snapshots are actual candidate
// values and can supply Read-/Read+/Full/Next without reference reconstruction.
BlockKVSequenceTape append_block_kv_sequence(DeviceProgram&,const weighted::KVConfig&,
  const WeightedRawState& initial,const at::Tensor& content,const at::Tensor& times,
  const at::Tensor& owners,const at::Tensor& present,const WeightedKVParameters&,
  const at::Tensor& error,int64_t tensor_budget);
// All roots must be present with their independently connected raw-field masks;
// unused cotangent/padding bytes may contain NaN. Every use of a historical
// write contributes, including connected-zero paths after complete eviction.
BlockKVSequenceGradient append_block_kv_sequence_vjp(DeviceProgram&,
  const BlockKVSequenceTape&,const BlockKVSequenceRoots&,const at::Tensor& error,int64_t tensor_budget);
}
