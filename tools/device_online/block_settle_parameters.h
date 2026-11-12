#pragma once
#include "block_settle_flow_vjp.h"
namespace tide::device_online {
std::map<std::string,WeightedRowAdjoint> block_kv_local_parameter_roots(const BlockKVLocalGradient&);
// The shared registry defines physical identity across every region and role.
// Names cover all uses of selected owners; numerical equality never merges them.
ParameterVjp append_block_settle_parameter_vjp(DeviceProgram&,const BlockSettleFlowGradient&,
  const ParameterRegistry&,const std::vector<BlockKVParameterRoles>&,
  const at::Tensor& error,int64_t tensor_budget);
// Parameter bank views in the same role vocabulary, for binding/publication.
std::map<std::string,at::Tensor> block_kv_local_parameters(const BlockKVLocalParameters&);
BlockKVLocalParameters block_kv_local_parameters(const std::map<std::string,at::Tensor>&,
  const BlockKVLocalParameters& shape_and_owner_binding);
}
