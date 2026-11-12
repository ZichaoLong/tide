#pragma once
#include "block_kv_bank.h"
namespace tide::device_online {
struct BlockKVReadPlan {
  BlockKVSnapshot snapshot;
  at::Tensor weights,decay; // [2,O,L,K], only true membership contributes.
  at::Tensor visible,ages,present; // [O,L,K], int64 ages, bool [O,L].
  at::Tensor query_owners,entry_owners,full_owners,clean; // Row-major Q, unique entries.
};
BlockKVReadPlan append_block_kv_read_plan(DeviceProgram&,const weighted::KVConfig&,const BlockKVSnapshot&,
  const at::Tensor& times,const at::Tensor& input,const at::Tensor& owners,const at::Tensor& present,
  int64_t parameter_banks,bool query,bool full,const at::Tensor& error,int64_t tensor_budget);
std::vector<at::Tensor> block_kv_read_plan_tensors(const BlockKVReadPlan&);
}
