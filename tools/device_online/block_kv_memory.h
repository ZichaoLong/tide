#pragma once
#include "block_kv_read_plan.h"
#include "block_kv_pool.h"
#include "block_kv_match.h"
#include "weighted_kv_query.h"
namespace tide::device_online {
struct BlockKVMemoryTape {
  BlockKVReadPlan plan;
  std::optional<WeightedProjectionTape> query,key,value;
  std::optional<BlockKVMatchTape> match;
  std::vector<BlockKVPoolTape> pools;
  at::Tensor memory,total; // Public [B*L*N,2,ds]/[B*L*N,2].
};
struct BlockKVMemoryGradient {
  BlockKVBankAdjoint bank;
  at::Tensor input,input_connected;
  WeightedKVQueryParameters parameters;
  at::Tensor query_connected,key_connected,value_connected;
};
BlockKVMemoryTape append_block_kv_memory(DeviceProgram&,const weighted::KVConfig&,const BlockKVSnapshot&,
  const at::Tensor& times,const at::Tensor& input,const at::Tensor& owners,const at::Tensor& present,
  const WeightedKVQueryParameters&,const at::Tensor& error,int64_t tensor_budget);
BlockKVMemoryGradient append_block_kv_memory_vjp(DeviceProgram&,const BlockKVMemoryTape&,
  const at::Tensor& memory_bar,const at::Tensor& total_bar,const at::Tensor& roots,
  const at::Tensor& error,int64_t tensor_budget);
std::vector<at::Tensor> block_kv_memory_tensors(const BlockKVMemoryTape&);
inline std::vector<at::Tensor> saved_tensors(const BlockKVMemoryTape& t){return block_kv_memory_tensors(t);}
// Shared Full/Read reverse helpers. No expanded floating raw snapshots.
struct BlockKVReadRoots {at::Tensor snapshot,query,keys,values;};
BlockKVReadRoots append_block_kv_read_roots(DeviceProgram&,const BlockKVReadPlan&,const at::Tensor& flags,
  int64_t kind,const at::Tensor& error);
at::Tensor append_block_kv_weight_vjp(DeviceProgram&,const BlockKVReadPlan&,const at::Tensor& bars,const at::Tensor& error);
}
