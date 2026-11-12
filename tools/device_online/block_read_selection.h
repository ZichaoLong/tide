#pragma once
#include "weighted_kv_query.h"
#include "weighted_kv_sum.h"
#include "weighted_describe.h"
#include "weighted_region.h"
#include "settle_block_plan.h"
#include "block_dispatch.h"
#include "tide/settle.h"
#include <optional>
#include <stdexcept>
#include <variant>

namespace tide::device_online {
// A closed Read/Describe -> complete-region SelStep invocation. R flattens B,L;
// node rows are [R,N], region rows [R,regions]. Snapshots are the caller's actual
// bound Read^0/Read^-/Read^+ operands, never inferred from output or reference.
struct BlockReadSelectionInput {
  WeightedRawState read;
  at::Tensor mass,content,present,times,node_owners;
};
struct BlockReadSelectionParameters {
  WeightedKVQueryParameters query; // Undefined in sum mode; no synthetic owners.
  WeightedDescribeParameters describe;
  WeightedRegionParameters region;
};
struct BlockReadSelectionTape {
  using Memory=std::variant<WeightedKVSumTape,WeightedKVQueryTape>;
  int64_t frames=0,nodes=0,regions=0,members=0;
  at::Tensor owners,present,node_times,order,inverse,ids,times,lengths,node_region_row;
  std::optional<Memory> memory;
  const Memory& memory_tape()const{
    if(!memory)throw std::invalid_argument("missing block Memory tape");return *memory;
  }
  const at::Tensor& memory_values()const{
    return std::visit([](const auto& t)->const at::Tensor&{return t.memory;},memory_tape());
  }
  const at::Tensor& memory_totals()const{
    return std::visit([](const auto& t)->const at::Tensor&{return t.total;},memory_tape());
  }
  WeightedDescribeTape describe;
  WeightedRegionTape region;
  at::Tensor controls,active; // [R,N], individual target support; no region-any replacement.
};
struct BlockReadSelectionGradient {
  at::Tensor raw,raw_connected,mass,mass_connected,content,content_connected;
  WeightedKVQueryParameters query;
  at::Tensor query_connected,key_connected,value_connected;
  std::map<std::string,WeightedRowAdjoint> describe;
  std::optional<WeightedHeadGradient> region_context,region_query;
};
// Independent observations, distinct from controls. Memory/total bars flatten
// B,L,N and use a separate bool per short/long channel; descriptions use one
// bool per row. An absent row is not a mathematical observation.
struct BlockReadObservations {
  WeightedRowAdjoint memory,total; // [R*N,2,ds]/[R*N,2], flags [R*N,2].
  std::map<std::string,WeightedRowAdjoint> describe;
};
BlockReadSelectionTape append_block_read_selection(DeviceProgram&,const SettleBlockPlan&,
  const weighted::KVConfig&,const weighted::RegionalConfig&,const BlockReadSelectionInput&,
  const BlockReadSelectionParameters&,const at::Tensor& error,int64_t tensor_budget);
BlockReadSelectionGradient append_block_read_selection_vjp(DeviceProgram&,const BlockReadSelectionTape&,
  const at::Tensor& controls_bar,const at::Tensor& roots,const at::Tensor& error,int64_t tensor_budget,
  const BlockReadObservations& = {});
SettleBlockPlan lower_settle_blocks(const SettleGraph&,int64_t first_position,
  int64_t batch_capacity,int64_t tensor_bytes);
void append_block_node_gate(DeviceProgram&,const BlockDispatcher&,const at::Tensor& node_active);
}
