#pragma once
// 0770593 weighted architecture §3 / §4.2: Read and Full have separate raw
// snapshots; messages carry (c,G), while cG is only a numerical projection.
#include "block_read_selection.h"
#include "weighted_kv_full.h"
#include "block_kv_full.h"
#include "weighted_parameter_vjp.h"

namespace tide::device_online {
struct BlockKVMessageInput {
  BlockReadSelectionInput read;
  WeightedRawState proposal; // Actual Upd result, never reconstructed from Read.
  at::Tensor full_owners; // Optional [R,N]; undefined reuses read.node_owners.
  std::optional<BlockKVSnapshot> proposal_bank;
};
struct BlockKVMessageParameters {
  BlockReadSelectionParameters read;
  WeightedKVFullParameters full;
};
struct BlockMessageColumns {
  at::Tensor strength,content,present,times; // [R,N], [R,N,d], bool/int64 [R,N].
  at::Tensor projection; // Optional cG [R,N,d]; never substituted for content.
};
struct BlockKVMessageTape {
  BlockReadSelectionTape selection;
  WeightedKVFullTape full;
  std::shared_ptr<WeightedRowGraph> columns;
  size_t strength_index,content_index;
  std::optional<size_t> projection_index;
  BlockMessageColumns messages;
  std::optional<BlockKVFullTape> bank_full;
};
// Each connected mask is bool [R,N]. Scalar bars are [R,N], vector bars
// [R,N,d]. Undefined value AND mask means no root. An unused bar may be NaN.
struct BlockKVMessageRoots {
  WeightedRowAdjoint controls,full,message_strength,message_content,projection;
  BlockReadObservations read;
};
struct BlockKVMessageGradient {
  BlockReadSelectionGradient read;
  WeightedKVFullGradient full; // proposal raw gradient is full.raw.
  WeightedRowAdjoint content; // Actual common input: read.content + full.input.
  std::optional<BlockKVBankAdjoint> full_bank;
};
BlockKVMessageTape append_block_kv_messages(DeviceProgram&,const SettleBlockPlan&,
  const weighted::KVConfig&,const weighted::RegionalConfig&,const BlockKVMessageInput&,
  const BlockKVMessageParameters&,const at::Tensor& error,int64_t tensor_budget,
  bool numerical_projection=false);
BlockKVMessageGradient append_block_kv_messages_vjp(DeviceProgram&,const BlockKVMessageTape&,
  const BlockKVMessageRoots&,const at::Tensor& error,int64_t tensor_budget);

// Explicit row-owner names in an existing ParameterRegistry. Keys are
// read.query/key/value, describe.<role>, region.context/query.<head-role>,
// full.query/out/gate/up/down. Mapping can select a trainable subset; every
// selected role needs one name per bank. Repeated names/aliases accumulate.
// Identity comes from the registry, never equality of independently packed data.
using BlockKVParameterRoles=std::map<std::string,std::vector<std::string>>;
std::map<std::string,WeightedRowAdjoint> block_kv_message_parameter_roots(const BlockKVMessageGradient&);
ParameterVjp append_block_kv_message_parameter_vjp(DeviceProgram&,const BlockKVMessageGradient&,
  const ParameterRegistry&,const BlockKVParameterRoles&,const at::Tensor& error,int64_t tensor_budget);
}
