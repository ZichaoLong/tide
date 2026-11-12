#pragma once
#include "block_kv_memory.h"
#include "weighted_kv_full.h"
namespace tide::device_online {
struct BlockKVFullTape {
  BlockKVReadPlan plan;
  WeightedProjectionTape query;
  std::vector<BlockKVMatchTape> matches;
  std::vector<BlockKVPoolTape> pools;
  WeightedResidualTape residual;
  at::Tensor output;
};
struct BlockKVFullGradient {BlockKVBankAdjoint bank;WeightedKVFullGradient rows;};
BlockKVFullTape append_block_kv_full(DeviceProgram&,const weighted::KVConfig&,const BlockKVSnapshot&,
  const at::Tensor& times,const at::Tensor& input,const at::Tensor& owners,const at::Tensor& active,
  const WeightedKVFullParameters&,const at::Tensor& error,int64_t tensor_budget);
BlockKVFullGradient append_block_kv_full_vjp(DeviceProgram&,const BlockKVFullTape&,
  const at::Tensor& gradient,const at::Tensor& roots,const at::Tensor& error,int64_t tensor_budget);
std::vector<at::Tensor> block_kv_full_tensors(const BlockKVFullTape&);
}
