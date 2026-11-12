#pragma once
#include "device_program.h"
namespace tide::device_online {
// Batched query x shared-entry pooling; values never expand over query rows.
struct BlockKVPoolTape {
  at::Tensor weights,values,visible,present,meta,probabilities,coefficients,total,mean,output;
  at::Tensor centered,anchor; // Shared [O,K,D]/[O,1,D], never query-expanded.
  bool attention=false;
};
struct BlockKVPoolGradient {at::Tensor weights,values,logits,connected;};
BlockKVPoolTape append_block_kv_pool(DeviceProgram&,const at::Tensor& weights,
  const at::Tensor& values,const at::Tensor& logits,const at::Tensor& visible,const at::Tensor& present,
  const at::Tensor& error,int64_t tensor_budget);
BlockKVPoolGradient append_block_kv_pool_vjp(DeviceProgram&,const BlockKVPoolTape&,
  const at::Tensor& vector_bar,const at::Tensor& total_bar,const at::Tensor& roots,
  const at::Tensor& error,int64_t tensor_budget);
std::vector<at::Tensor> block_kv_pool_tensors(const BlockKVPoolTape&);
}
