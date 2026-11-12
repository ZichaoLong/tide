#pragma once
// Fixed weighted architecture §2–4: complete fibers -> actual KV sequence ->
// complete region -> structured messages. No per-event host numerical calls.
#include "block_kv_sequence.h"
#include "block_kv_messages.h"
#include "weighted_aggregate.h"
#include "weighted_source_adaptation.h"
namespace tide::device_online {
struct BlockKVLocalInput {
  at::Tensor payload,lengths,source_owners; // [B,L,N,A,d+1], int64 [B,L,N]/[B,L,N,A]
  at::Tensor times,node_owners; // int64 [B,L,N]
  WeightedRawState initial,bound_initial; // [B*N,*]; Read0 has its own binding.
};
struct BlockKVLocalParameters {
  WeightedSourceParameters source;
  WeightedKVParameters update;
  BlockKVMessageParameters message;
  std::optional<WeightedSourceFactors> source_factors;
};
struct BlockKVLocalTape {
  std::string read_mode;
  WeightedAggregateTape aggregate;
  BlockKVSequenceTape sequence;
  BlockKVMessageTape messages;
  WeightedRawState read;
  at::Tensor initial_mapping,present;
  std::shared_ptr<WeightedRowGraph> received;
  size_t received_index,mass_index,content_index;
  std::optional<WeightedSourceAdaptation> source_adaptation;
};
struct BlockKVLocalRoots {
  BlockKVMessageRoots messages;
  WeightedRowAdjoint received; // [B*L*N,d+1], one bool per candidate.
  BlockKVSequenceRoots raw; // independently rooted old/proposal/final.
};
struct BlockKVLocalGradient {
  WeightedAggregateGradient aggregate; // Physical atoms, never pre-multiplied cG.
  BlockKVSequenceGradient sequence;
  BlockKVMessageGradient messages;
  at::Tensor bound_initial,bound_initial_connected;
  std::optional<WeightedSourceAdaptationGradient> source_adaptation;
};
BlockKVLocalTape append_block_kv_local(DeviceProgram&,const SettleBlockPlan&,
  const weighted::KVConfig&,const weighted::RegionalConfig&,const std::string& read_mode,
  float aggregate_alpha,const BlockKVLocalInput&,const BlockKVLocalParameters&,
  const at::Tensor& error,int64_t tensor_budget,bool numerical_projection=false);
BlockKVLocalGradient append_block_kv_local_vjp(DeviceProgram&,const BlockKVLocalTape&,
  const BlockKVLocalRoots&,const at::Tensor& error,int64_t tensor_budget);
}
