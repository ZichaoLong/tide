#pragma once
// KV §§1.3/3/4: one numerical entry bank, independently named raw snapshots.
#include "block_kv_sequence.h"
namespace tide::device_online {
struct BlockKVBank {
  int64_t batch=0,length=0,nodes=0,capacity=0,entries=0;
  WeightedRawLayout layout;
  WeightedRawState initial;
  at::Tensor keys,values,summaries,weights,times,valid; // [B*N,C+L,width], int64/bool metadata.
  at::Tensor old_mapping,proposal_mapping; // [B*L*N,C], global entry ids; -1 padding.
  at::Tensor old_integer,proposal_integer; // Actual raw clocks/counts/timestamps.
  at::Tensor writes; // bool [B*L*N], independently of candidate/Full activity.
  at::Tensor final_mapping;
};
struct BlockKVSnapshot {
  std::shared_ptr<BlockKVBank> bank;
  at::Tensor mapping,integer; // [rows,C]/[rows,4+C]; indices refer to actual bank values.
  // Static row layout: old/proposal have B,L,N; final has B,1,N.
  int64_t length=0;
  int64_t prefix_offset=0; // old:0, proposal:1; Read0 has no current writes.
};
struct BlockKVBankAdjoint {
  at::Tensor keys,values,summaries,weights; // Same numerical bank extents.
  at::Tensor snapshot_connected; // [B*L*N,5], causal raw fields, not entry visibility.
  BlockKVSnapshot snapshot;
};
BlockKVSequenceTape append_block_kv_bank_sequence(DeviceProgram&,const weighted::KVConfig&,
  const WeightedRawState&,const at::Tensor& content,const at::Tensor& times,const at::Tensor& owners,
  const at::Tensor& present,const WeightedKVParameters&,const at::Tensor& error,int64_t tensor_budget);
BlockKVSnapshot block_kv_snapshot(const BlockKVSequenceTape&,bool proposal);
BlockKVSnapshot block_kv_final_snapshot(const BlockKVSequenceTape&);
BlockKVSnapshot append_block_kv_bound_bank(DeviceProgram&,const weighted::KVConfig&,const WeightedRawState&,
  int64_t batch,int64_t length,int64_t nodes,const at::Tensor& error,int64_t tensor_budget);
struct BlockKVBankRawRoot {BlockKVSnapshot snapshot;at::Tensor values,connected;};
struct BlockKVBankGradient {at::Tensor initial,initial_connected,writes,write_connected;};
BlockKVBankGradient append_block_kv_bank_vjp(DeviceProgram&,const std::shared_ptr<BlockKVBank>&,
  const std::vector<BlockKVBankAdjoint>&,const std::vector<BlockKVBankRawRoot>&,
  const at::Tensor& error,int64_t tensor_budget);
// Explicit observation/continuation decoder; never called by bank Mem/Full.
WeightedRawState append_block_kv_bank_decode(DeviceProgram&,const BlockKVSnapshot&,
  const at::Tensor& error,int64_t tensor_budget);
std::vector<at::Tensor> block_kv_bank_tensors(const BlockKVBank&);
}
