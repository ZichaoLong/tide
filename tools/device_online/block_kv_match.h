#pragma once
#include "device_program.h"
namespace tide::device_online {
struct BlockKVMatchTape {at::Tensor query,keys,ages,visible,phase,cosine,sine,scale,logits;int64_t pairs=0;};
struct BlockKVMatchGradient {at::Tensor query,keys;};
BlockKVMatchTape append_block_kv_match(DeviceProgram&,const at::Tensor& query,const at::Tensor& keys,
  const at::Tensor& ages,const at::Tensor& visible,double position_scale,const std::vector<double>& frequencies,
  double distance_slope,const at::Tensor& error,int64_t tensor_budget);
BlockKVMatchGradient append_block_kv_match_vjp(DeviceProgram&,const BlockKVMatchTape&,
  const at::Tensor& logits_bar,const at::Tensor& error,int64_t tensor_budget);
std::vector<at::Tensor> block_kv_match_tensors(const BlockKVMatchTape&);
}
